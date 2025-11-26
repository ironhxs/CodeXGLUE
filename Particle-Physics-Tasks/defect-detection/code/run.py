#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缺陷检测任务 - 训练和评估脚本
基于 CodeXGLUE Defect-detection 任务改编
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

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import DefectDetectionModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """单个样本的特征"""
    def __init__(self, input_ids, attention_mask, label, idx):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.idx = idx


def convert_examples_to_features(js, tokenizer, args):
    """将单个样本转换为特征"""
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    
    # Padding
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * (args.block_size - padding_length) + [0] * padding_length
    
    return InputFeatures(
        input_ids=source_ids,
        attention_mask=attention_mask,
        label=js['target'],
        idx=js.get('idx', 0)
    )


class DefectDataset(Dataset):
    """缺陷检测数据集"""
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
        
        # 统计标签分布
        labels = [ex.label for ex in self.examples]
        logger.info(f"  Label 0 (no defect): {labels.count(0)}")
        logger.info(f"  Label 1 (has defect): {labels.count(1)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.examples[idx].input_ids),
            torch.tensor(self.examples[idx].attention_mask),
            torch.tensor(self.examples[idx].label),
        )


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
                                  batch_size=args.train_batch_size)
    
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
    
    best_f1 = 0
    
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        total_loss = 0
        
        for step, batch in enumerate(bar):
            input_ids, attention_mask, labels = [x.to(args.device) for x in batch]
            
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
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
        
        # 评估
        if args.do_eval:
            results = evaluate(args, model, tokenizer)
            logger.info(f"Epoch {epoch} - F1: {results['f1']:.4f}, Acc: {results['accuracy']:.4f}")
            
            if results['f1'] > best_f1:
                best_f1 = results['f1']
                output_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
                logger.info(f"  Best model saved with F1: {best_f1:.4f}")


def evaluate(args, model, tokenizer):
    """评估模型"""
    eval_dataset = DefectDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                                 batch_size=args.eval_batch_size)
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, attention_mask, labels = [x.to(args.device) for x in batch]
        
        with torch.no_grad():
            prob = model(input_ids=input_ids, attention_mask=attention_mask)
        
        preds = torch.argmax(prob, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    model.train()
    
    # 计算指标
    results = compute_metrics(all_preds, all_labels)
    return results


def test(args, model, tokenizer):
    """测试模型"""
    test_dataset = DefectDataset(tokenizer, args, args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.eval_batch_size)
    
    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids, attention_mask, labels = [x.to(args.device) for x in batch]
        
        with torch.no_grad():
            prob = model(input_ids=input_ids, attention_mask=attention_mask)
        
        preds = torch.argmax(prob, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(prob[:, 1].cpu().numpy().tolist())  # 正类概率
    
    # 保存预测结果
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_preds:
            f.write(str(pred) + '\n')
    
    gold_file = os.path.join(args.output_dir, "gold.txt")
    with open(gold_file, 'w', encoding='utf-8') as f:
        for label in all_labels:
            f.write(str(label) + '\n')
    
    prob_file = os.path.join(args.output_dir, "probabilities.txt")
    with open(prob_file, 'w', encoding='utf-8') as f:
        for prob in all_probs:
            f.write(f"{prob:.4f}\n")
    
    # 计算指标
    results = compute_metrics(all_preds, all_labels)
    
    logger.info("Test Results:")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1:        {results['f1']:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return results


def compute_metrics(predictions, labels):
    """计算分类指标"""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--block_size", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    
    # 训练参数
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
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
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    
    model = DefectDetectionModel(encoder, config, tokenizer, args)
    model.to(args.device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info(f"Model loaded: {args.model_name_or_path}")
    
    # 训练
    if args.do_train:
        train_dataset = DefectDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)
    
    # 测试
    if args.do_test:
        checkpoint = os.path.join(args.output_dir, 'best_model', 'model.bin')
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint))
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
