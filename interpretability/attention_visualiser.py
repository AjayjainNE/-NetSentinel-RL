"""Attention weight extraction and visualisation from DistilBERT classifier."""
import numpy as np
import torch
from typing import List, Dict, Tuple
import logging
log = logging.getLogger(__name__)

class AttentionVisualiser:
    """Extracts and processes attention weights from HF transformer models."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_attention_weights(self, text: str) -> Tuple[List[str], np.ndarray]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # Stack all layers, average over heads: shape (n_layers, seq_len, seq_len)
        attn = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        attn = attn.squeeze(1).mean(dim=1)       # (layers, seq, seq)
        attn_mean = attn.mean(dim=0).numpy()     # (seq, seq) — layer-averaged
        return tokens, attn_mean

    def top_attended_tokens(self, text: str, top_k: int = 5) -> List[Dict]:
        tokens, attn = self.get_attention_weights(text)
        # Use [CLS] token row as global importance
        cls_attn = attn[0]
        indices = np.argsort(cls_attn)[::-1][:top_k]
        return [{"token": tokens[i], "attention": float(cls_attn[i])} for i in indices if tokens[i] not in ["[CLS]","[SEP]","[PAD]"]]

    def flow_token_importance(self, flow_text: str) -> Dict[str, float]:
        """Map flow field names to their attention importance."""
        tokens, attn = self.get_attention_weights(flow_text)
        cls_attn = attn[0]
        importance = {}
        field_keywords = ["duration","packets","bytes","flags","syn","rst","ack","iat","ratio","size","psh","urg"]
        for i, tok in enumerate(tokens):
            for kw in field_keywords:
                if kw in tok.lower():
                    importance[tok] = importance.get(tok,0) + float(cls_attn[i])
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
