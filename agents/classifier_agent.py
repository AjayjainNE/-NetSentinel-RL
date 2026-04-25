"""
NetSentinel-RL — Classifier Agent
Fine-tunes DistilBERT on flow-as-text representations with RL reward shaping.

Network flow statistics are serialised to structured natural-language tokens,
enabling pre-trained language-model transfer while an asymmetric RL reward
signal steers training toward the operational F1 objective.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import evaluate

log = logging.getLogger(__name__)

ATTACK_NAMES = [
    "Benign", "DoS", "DDoS", "PortScan",
    "BruteForce", "Botnet", "WebAttack", "Infiltration",
]
N_CLASSES = len(ATTACK_NAMES)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FlowTextDataset(Dataset):
    """Flow text representations paired with integer class labels."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.labels    = torch.tensor(labels, dtype=torch.long)
        log.info("Tokenising %s flows …", f"{len(texts):,}")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Custom trainer with RL reward shaping ─────────────────────────────────────

class RewardShapedTrainer(Trainer):
    """
    Hugging Face Trainer with an asymmetric RL-inspired loss term.

    Standard cross-entropy is augmented by:
    - A false-negative penalty: the model is penalised when it assigns
      high benign probability to actual attack flows.  The penalty weight
      (FN_WEIGHT = 2.0) mirrors the FN_COST / TP_REWARD ratio in NetworkEnv.
    - A smaller false-positive penalty for high attack probability on benign
      flows, matching the FP_COST / TP_REWARD ratio (FP_WEIGHT = 0.8).

    This bridges the gap between the supervised cross-entropy objective and
    the operational metric (weighted F1 with asymmetric costs).
    """

    FN_WEIGHT: float = 2.0
    FP_WEIGHT: float = 0.8

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels  = inputs["labels"]
        outputs = model(**inputs)
        logits  = outputs.logits

        ce_loss  = nn.CrossEntropyLoss()(logits, labels)
        probs    = torch.softmax(logits, dim=-1)
        benign_p = probs[:, 0]

        is_attack = (labels != 0).float()
        is_benign = (labels == 0).float()

        fn_penalty = (is_attack * benign_p).mean()
        fp_penalty = (is_benign * (1.0 - benign_p)).mean()

        loss = ce_loss + self.FN_WEIGHT * fn_penalty + self.FP_WEIGHT * fp_penalty
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: EvalPrediction) -> dict:
    f1_metric  = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    preds  = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    results = {
        "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        "accuracy":    acc_metric.compute(predictions=preds, references=labels)["accuracy"],
    }

    per_class = f1_metric.compute(predictions=preds, references=labels, average=None)
    if per_class and "f1" in per_class:
        for i, name in enumerate(ATTACK_NAMES):
            if i < len(per_class["f1"]):
                results[f"f1_{name}"] = round(float(per_class["f1"][i]), 4)

    return results


# ── Build / train / load ──────────────────────────────────────────────────────

def build_classifier(model_name: str = "distilbert-base-uncased") -> tuple:
    """Return (tokenizer, model) for fine-tuning."""
    log.info("Loading %s for %d-class classification …", model_name, N_CLASSES)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=N_CLASSES,
        id2label={i: n for i, n in enumerate(ATTACK_NAMES)},
        label2id={n: i for i, n in enumerate(ATTACK_NAMES)},
    )
    return tokenizer, model


def train_classifier(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_val: list[str],
    y_val: np.ndarray,
    model_name: str = "distilbert-base-uncased",
    save_path: str = "models/classifier",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    use_reward_shaping: bool = True,
) -> tuple:
    """Fine-tune DistilBERT; returns (trainer, tokenizer, model)."""

    tokenizer, model = build_classifier(model_name)
    train_ds = FlowTextDataset(texts_train, y_train, tokenizer)
    val_ds   = FlowTextDataset(texts_val,   y_val,   tokenizer)

    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=f"{save_path}/logs",
        logging_steps=100,
        report_to=["mlflow"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    TrainerCls = RewardShapedTrainer if use_reward_shaping else Trainer
    trainer    = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    label = "reward-shaped" if use_reward_shaping else "standard"
    log.info("Training classifier (%s) …", label)
    trainer.train()

    metrics = trainer.evaluate()
    log.info("Validation metrics: %s", metrics)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    log.info("Classifier saved → %s", save_path)
    return trainer, tokenizer, model


def load_classifier(save_path: str = "models/classifier") -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model     = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.eval()
    return tokenizer, model


def predict(
    flow_text: str,
    tokenizer,
    model,
    return_probs: bool = False,
) -> dict:
    """Run inference on a single serialised flow string."""
    inputs = tokenizer(flow_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_class = int(np.argmax(probs))
    result = {
        "predicted_class": pred_class,
        "predicted_label": ATTACK_NAMES[pred_class],
        "confidence":      float(probs[pred_class]),
    }
    if return_probs:
        result["class_probs"] = {n: float(p) for n, p in zip(ATTACK_NAMES, probs)}
    return result
