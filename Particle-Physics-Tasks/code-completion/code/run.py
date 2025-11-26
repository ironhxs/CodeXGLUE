#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码补全任务 - 训练和评估脚本
基于 CodeXGLUE CodeCompletion-line 任务改编
"""

from __future__ import absolute_import
import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
from io import open
from tqdm import tqdm
from fuzzywuzzy import fuzz

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from model import CodeCompletionModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """单个样本"""
    def __init__(self, idx, context, target):
        self.idx = idx
        self.context = context
        self.target = target


def read_examples(filename):
    """从 JSONL 文件读取样本"""
    examples = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            
            context = js.get('context', '')
            target = js.get('target', '')
            
            if context and target:
                examples.append(Example(idx=idx, context=context, target=target))
    
    return examples


class CompletionDataset(Dataset):
    """代码补全数据集"""
    def __init__(self, examples, tokenizer, args, is_train=True):
        self.examples = []
        self.tokenizer = tokenizer
        self.args = args
        
        for ex in tqdm(examples, desc="Processing examples"):
            # 编码上下文
            context_tokens = tokenizer.encode(ex.context, add_special_tokens=False)
            target_tokens = tokenizer.encode(ex.target, add_special_tokens=False)
            
            # 拼接并截断
            max_len = args.block_size
            combined = context_tokens + target_tokens
            
            if len(combined) > max_len:
                # 保留上下文尾部和目标
                context_len = max_len - len(target_tokens) - 1
                if context_len > 10:
                    context_tokens = context_tokens[-context_len:]
                    combined = context_tokens + target_tokens
                else:
                    continue
            
            # 创建输入和标签
            input_ids = combined
            
            if is_train:
                # 训练时，标签是整个序列
                labels = [-100] * len(context_tokens) + target_tokens
            else:
                # 评估时，只需要上下文
                input_ids = context_tokens
                labels = target_tokens
            
            # padding
            padding_length = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            
            if is_train:
                labels = labels + [-100] * padding_length
            
            self.examples.append({
                'input_ids': input_ids[:max_len],
                'labels': labels if is_train else target_tokens,
                'context': ex.context,
                'target': ex.target,
                'context_len': len(context_tokens),
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """批处理函数"""
    input_ids = torch.tensor([ex['input_ids'] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex['labels'] for ex in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': labels}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, train_dataset, model, tokenizer):
    """训练模型"""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, 
                                  collate_fn=collate_fn)
    
    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                                                 num_training_steps=t_total)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    
    model.zero_grad()
    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        total_loss = 0
        
        for step, batch in enumerate(bar):
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            total_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # 每个 epoch 评估
        if args.do_eval:
            results = evaluate(args, model, tokenizer)
            logger.info(f"Epoch {epoch} - Edit Sim: {results['edit_sim']:.2f}, EM: {results['exact_match']:.2f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                output_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(output_dir, exist_ok=True)
                model.model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"  Best model saved with loss: {best_loss:.4f}")


def evaluate(args, model, tokenizer):
    """评估模型"""
    eval_examples = read_examples(args.eval_data_file)
    eval_dataset = CompletionDataset(eval_examples, tokenizer, args, is_train=False)
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    
    model.eval()
    predictions = []
    references = []
    
    for ex in tqdm(eval_dataset.examples, desc="Evaluating"):
        context = ex['context']
        target = ex['target']
        
        input_ids = tokenizer.encode(context, return_tensors='pt').to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=input_ids.size(1) + args.max_gen_length,
                num_beams=args.beam_size,
                do_sample=False,
            )
        
        # 提取生成的部分
        generated = outputs[0][input_ids.size(1):]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        
        predictions.append(pred.strip())
        references.append(target.strip())
    
    model.train()
    
    # 计算指标
    results = compute_metrics(predictions, references)
    return results


def test(args, model, tokenizer):
    """测试模型"""
    test_examples = read_examples(args.test_data_file)
    test_dataset = CompletionDataset(test_examples, tokenizer, args, is_train=False)
    
    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    
    model.eval()
    predictions = []
    references = []
    
    for ex in tqdm(test_dataset.examples, desc="Testing"):
        context = ex['context']
        target = ex['target']
        
        input_ids = tokenizer.encode(context, return_tensors='pt').to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=input_ids.size(1) + args.max_gen_length,
                num_beams=args.beam_size,
                do_sample=False,
            )
        
        generated = outputs[0][input_ids.size(1):]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        
        predictions.append(pred.strip())
        references.append(target.strip())
    
    # 保存预测结果
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred.replace('\n', ' ') + '\n')
    
    gold_file = os.path.join(args.output_dir, "gold.txt")
    with open(gold_file, 'w', encoding='utf-8') as f:
        for ref in references:
            f.write(ref.replace('\n', ' ') + '\n')
    
    # 计算指标
    results = compute_metrics(predictions, references)
    
    logger.info(f"Test Results:")
    logger.info(f"  Edit Similarity: {results['edit_sim']:.2f}")
    logger.info(f"  Exact Match: {results['exact_match']:.2f}")
    
    return results


def compute_metrics(predictions, references):
    """计算评估指标"""
    edit_similarities = []
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        # 编辑距离相似度 (使用 fuzzywuzzy)
        edit_sim = fuzz.ratio(pred, ref)
        edit_similarities.append(edit_sim)
        
        # 精确匹配
        if pred.strip() == ref.strip():
            exact_matches += 1
    
    return {
        'edit_sim': sum(edit_similarities) / len(edit_similarities) if edit_similarities else 0,
        'exact_match': exact_matches / len(predictions) * 100 if predictions else 0,
        'num_samples': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    
    # 模型参数
    parser.add_argument("--model_name_or_path", default="microsoft/CodeGPT-small-py", type=str,
                        help="预训练模型: microsoft/CodeGPT-small-py, gpt2, etc.")
    parser.add_argument("--block_size", default=512, type=int)
    parser.add_argument("--max_gen_length", default=50, type=int,
                        help="最大生成长度")
    parser.add_argument("--beam_size", default=5, type=int)
    
    # 训练参数
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    logger.info(f"Device: {args.device}, n_gpu: {args.n_gpu}")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    try:
        config = GPT2Config.from_pretrained(args.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        gpt_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    except:
        # 使用 AutoModel 作为后备
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        gpt_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        config = gpt_model.config
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = CodeCompletionModel(gpt_model, config, tokenizer, args)
    model.to(args.device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info(f"Model loaded: {args.model_name_or_path}")
    
    # 训练
    if args.do_train:
        train_examples = read_examples(args.train_data_file)
        train_dataset = CompletionDataset(train_examples, tokenizer, args, is_train=True)
        train(args, train_dataset, model, tokenizer)
    
    # 测试
    if args.do_test:
        checkpoint = os.path.join(args.output_dir, 'best_model')
        if os.path.exists(checkpoint):
            gpt_model = GPT2LMHeadModel.from_pretrained(checkpoint)
            model = CodeCompletionModel(gpt_model, config, tokenizer, args)
            model.to(args.device)
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
