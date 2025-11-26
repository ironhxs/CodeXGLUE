# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
代码补全模型 - 基于 GPT-2/CodeGPT
"""

import torch
import torch.nn as nn


class CodeCompletionModel(nn.Module):
    """
    代码补全模型
    基于 GPT-2 的因果语言模型
    """
    def __init__(self, model, config, tokenizer, args):
        super(CodeCompletionModel, self).__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
    
    def forward(self, input_ids, labels=None):
        """
        前向传播
        Args:
            input_ids: 输入 token ids [batch_size, seq_len]
            labels: 标签 (用于计算损失) [batch_size, seq_len]
        """
        outputs = self.model(input_ids, labels=labels)
        return outputs
    
    def generate(self, input_ids, max_length=50, num_beams=1, 
                 do_sample=False, temperature=1.0, top_k=50, top_p=0.95):
        """
        生成代码补全
        """
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return outputs
