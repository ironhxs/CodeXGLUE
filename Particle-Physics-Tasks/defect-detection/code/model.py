# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
缺陷检测模型 - 基于 CodeBERT 的二分类
"""

import torch
import torch.nn as nn


class DefectDetectionModel(nn.Module):
    """
    缺陷检测模型
    基于 CodeBERT 的二分类器
    """
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectDetectionModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        # 分类头
        self.dropout = nn.Dropout(args.dropout if hasattr(args, 'dropout') else 0.1)
        self.classifier = nn.Linear(config.hidden_size, 2)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        前向传播
        Args:
            input_ids: 输入 token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 0/1 [batch_size]
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # 使用 [CLS] token 的表示
        pooled_output = outputs[0][:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, 2]
        
        prob = torch.softmax(logits, dim=-1)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
