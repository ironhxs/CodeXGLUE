# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
代码审查意图分类模型
基于 CodeBERT/RoBERTa 的 4 分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Model(nn.Module):   
    """代码审查意图分类模型"""
    
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        # 4 分类: 优化建议/错误报告/澄清请求/批准通过
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids=None, labels=None): 
        """
        Args:
            input_ids: [batch_size, seq_len] - tokenized input
            labels: [batch_size] - intent labels (0-3)
        
        Returns:
            loss (if labels provided), logits, probabilities
        """
        # 获取编码器输出
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=input_ids.ne(1)  # padding token id = 1
        )
        
        # 使用 [CLS] token 的表示（第一个 token）
        pooled_output = outputs[0][:, 0, :]
        
        # Dropout + 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 如果提供了标签，计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, probs
        else:
            return probs
