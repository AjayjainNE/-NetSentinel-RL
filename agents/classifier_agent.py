"""
NetSentinel-RL — Classifier Agent
Fine-tunes DistilBERT on flow-as-text representations with RL reward shaping.

Novel approach: network flow statistics are converted to structured natural
language tokens, enabling transfer learning from pre-trained LMs while adding
an RL reward signal on top of standard cross-entropy loss.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import evaluate
from pathlib import Path
from typing import Optional
import logging

log = logging.getLogger(__name__)

# Attack class names matching environment/network_env.py ATTACK_NAMES
ATTACK_NAMES = [
    "Benign", "DoS", "DDoS", "PortScan",
    "BruteForce", "Botnet", "WebAttack", "Infiltration"
]


class FlowTextDataset(Dataset):
    """
    Dataset of flow text representations with integer labels.
    Uses FlowTextEncoder output from data/preprocess.py.
    """

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.labels = torch.tensor(labels, dtype=torch.long)
        log.info(f"Tokenising {len(texts):,} flow texts...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


class RLRewardShapedTrainer(Trainer):
    """
    Custom HuggingFace Trainer that augments the cross-entropy loss
    with an RL-inspired reward signal.

    Reward shaping strategy:
    - Standard cross-entropy provides the base gradient
    - An additional penalty term is added when the model confidently
      misclassifies attacks as benign (false negatives), mimicking
      the FN_COST in the RL environment
    - A smaller penalty for false positives (benign as attack)

    This bridges the gap between supervised learning objectives
    (accuracy) and the operational metric we actually care about (F1
    with asymmetric FP/FN costs).
    """

    FN_PENALTY_WEIGHT = 2.0   # Matches NetworkEnv.FN_COST / TP_REWARD ratio
    FP_PENALTY_WEIGHT = 0.8

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard cross-entropy
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        # RL reward shaping: compute penalty for high-confidence errors
        probs = torch.softmax(logits, dim=-1)
        benign_prob = probs[:, 0]

        is_attack = (labels != 0).float()
        is_benign = (labels == 0).float()

        # FN penalty: model assigns high probability to benign for actual attacks
        fn_penalty = (is_attack * benign_prob).mean()

        # FP penalty: model assigns low probability to benign for actual benign flows
        fp_penalty = (is_benign * (1 - benign_prob)).mean()

        total_loss = (
            ce_loss
            + self.FN_PENALTY_WEIGHT * fn_penalty
            + self.FP_PENALTY_WEIGHT * fp_penalty
        )

        return (total_loss, outputs) if return_outputs else total_loss


def compute_metrics(pred: EvalPrediction) -> dict:
    """Compute F1, precision, recall for HF Trainer evaluation."""
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    f1_result = metric_f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    acc_result = metric_acc.compute(predictions=predictions, references=labels)

    # Per-class F1 for interpretability
    per_class_f1 = metric_f1.compute(
        predictions=predictions, references=labels, average=None
    )

    results = {
        "f1_weighted": f1_result["f1"],
        "accuracy":    acc_result["accuracy"],
    }
    if per_class_f1 and "f1" in per_class_f1:
        for i, name in enumerate(ATTACK_NAMES):
            if i < len(per_class_f1["f1"]):
                results[f"f1_{name}"] = round(float(per_class_f1["f1"][i]), 4)
    return results


def build_classifier(
    model_name: str = "distilbert-base-uncased",
    n_labels: int = 8,
) -> tuple:
    """Load tokeniser and model for sequence classification."""
    log.info(f"Loading {model_name} for {n_labels}-class classification...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
        id2label={i: name for i, name in enumerate(ATTACK_NAMES)},
        label2id={name: i for i, name in enumerate(ATTACK_NAMES)},
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
    use_rl_shaping: bool = True,
) -> tuple:
    """
    Fine-tune DistilBERT on flow text data with optional RL reward shaping.
    Returns (trainer, tokenizer, model).
    """
    tokenizer, model = build_classifier(model_name, n_labels=len(ATTACK_NAMES))

    train_dataset = FlowTextDataset(texts_train, y_train, tokenizer)
    val_dataset   = FlowTextDataset(texts_val,   y_val,   tokenizer)

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=f"{save_path}/logs",
        logging_steps=100,
        report_to=["mlflow"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    TrainerClass = RLRewardShapedTrainer if use_rl_shaping else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    log.info(f"Training classifier ({'RL-shaped' if use_rl_shaping else 'standard'})...")
    trainer.train()

    metrics = trainer.evaluate()
    log.info(f"Final val metrics: {metrics}")

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    log.info(f"Classifier saved to {save_path}")

    return trainer, tokenizer, model


def load_classifier(save_path: str = "models/classifier"):
    """Load a saved classifier for inference."""
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.eval()
    return tokenizer, model


def predict_threat_class(
    flow_text: str,
    tokenizer,
    model,
    return_probs: bool = False,
) -> dict:
    """Run inference on a single flow text representation."""
    inputs = tokenizer(
        flow_text, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    pred_class = int(np.argmax(probs))

    result = {
        "predicted_class": pred_class,
        "predicted_label": ATTACK_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
    }
    if return_probs:
        result["class_probs"] = {name: float(p) for name, p in zip(ATTACK_NAMES, probs)}
    return result
