"""
Model Definition for Violation Classifier
==========================================
RoBERTa 기반 대화 위반 탐지 분류기 모델
"""

import torch
import torch.nn as nn
from transformers import RobertaModel


class ViolationClassifier(nn.Module):
    """
    RoBERTa 기반 위반 분류기
    
    구조:
        RoBERTa Encoder → Pooling ([CLS] or Mean) → Dropout → Linear → Logits
    
    Args:
        model_name (str): RoBERTa 모델 이름 (예: "roberta-base")
        num_labels (int): 클래스 개수
        dropout (float): Dropout 비율
        pooling (str): "cls" (CLS 토큰) or "mean" (평균 풀링)
    """
    
    def __init__(self, model_name, num_labels, dropout=0.1, pooling="cls"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): [batch_size, seq_len]
            attention_mask (torch.Tensor): [batch_size, seq_len]
        
        Returns:
            logits (torch.Tensor): [batch_size, num_labels]
        """
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pooling
        if self.pooling == "cls":
            # [CLS] token pooling (기본 권장)
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling (attention mask 고려)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Dropout + Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
