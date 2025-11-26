#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码摘要生成任务 - 训练和评估脚本
基于 CodeXGLUE code-to-text 任务改编
"""

from __future__ import absolute_import
import os
import sys
import json
import pickle
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
    RobertaTokenizer
)

from model import Seq2Seq

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """单个训练/测试样本"""
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """从 JSONL 文件读取样本"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            
            # 处理代码 diff
            code = js.get('code', '')
            code = ' '.join(code.split())[:2000]
            
            # 处理摘要 (MR title)
            summary = js.get('summary', '')
            summary = ' '.join(summary.split())
            
            examples.append(Example(idx=idx, source=code, target=summary))
    
    return examples


class InputFeatures(object):
    """单个样本的特征"""
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    """将样本转换为模型输入特征"""
    features = []
    
    for example_index, example in enumerate(tqdm(examples, desc="Converting examples")):
        # 源序列 (代码 diff)
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        
        # 目标序列 (摘要)
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        
        features.append(InputFeatures(
            example_id=example_index,
            source_ids=source_ids,
            target_ids=target_ids,
            source_mask=source_mask,
            target_mask=target_mask,
        ))
    
    return features


class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return (
            torch.tensor(self.examples[item].source_ids),
            torch.tensor(self.examples[item].source_mask),
            torch.tensor(self.examples[item].target_ids),
            torch.tensor(self.examples[item].target_mask),
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
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    
    model.zero_grad()
    model.train()
    
    best_bleu = 0
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(bar):
            source_ids, source_mask, target_ids, target_mask = [x.to(args.device) for x in batch]
            
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            bar.set_postfix(loss=loss.item())
        
        # 每个 epoch 结束后评估
        if args.do_eval:
            results = evaluate(args, model, tokenizer)
            logger.info(f"Epoch {epoch} - BLEU: {results['bleu']:.4f}")
            
            if results['bleu'] > best_bleu:
                best_bleu = results['bleu']
                output_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
                logger.info(f"  Best model saved with BLEU: {best_bleu:.4f}")


def evaluate(args, model, tokenizer):
    """评估模型"""
    eval_examples = read_examples(args.eval_data_file)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
    eval_dataset = TextDataset(eval_features, args)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    
    model.eval()
    predictions = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        source_ids, source_mask, target_ids, target_mask = [x.to(args.device) for x in batch]
        
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if tokenizer.eos_token_id in t:
                    t = t[:t.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(t, skip_special_tokens=True)
                predictions.append(text)
    
    model.train()
    
    # 计算 BLEU
    references = [ex.target for ex in eval_examples]
    bleu_score = compute_bleu(predictions, references)
    
    return {'bleu': bleu_score, 'predictions': predictions}


def test(args, model, tokenizer):
    """测试模型"""
    test_examples = read_examples(args.test_data_file)
    test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
    test_dataset = TextDataset(test_features, args)
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    
    model.eval()
    predictions = []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        source_ids, source_mask, target_ids, target_mask = [x.to(args.device) for x in batch]
        
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if tokenizer.eos_token_id in t:
                    t = t[:t.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(t, skip_special_tokens=True)
                predictions.append(text)
    
    # 保存预测结果
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred.strip() + '\n')
    
    # 保存真实标签
    gold_file = os.path.join(args.output_dir, "gold.txt")
    with open(gold_file, 'w', encoding='utf-8') as f:
        for ex in test_examples:
            f.write(ex.target.strip() + '\n')
    
    # 计算 BLEU
    references = [ex.target for ex in test_examples]
    bleu_score = compute_bleu(predictions, references)
    
    logger.info(f"Test BLEU: {bleu_score:.4f}")
    
    return {'bleu': bleu_score}


def compute_bleu(predictions, references):
    """计算 BLEU 分数"""
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def sentence_bleu(pred, ref, max_n=4):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(overlap / total)
        
        # 几何平均
        if min(precisions) > 0:
            log_avg = sum(math.log(p) for p in precisions) / len(precisions)
            geo_mean = math.exp(log_avg)
        else:
            geo_mean = 0.0
        
        # 简短惩罚
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        
        return bp * geo_mean
    
    scores = [sentence_bleu(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores) * 100 if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="训练数据文件路径")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="验证数据文件路径")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="测试数据文件路径")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型输出目录")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="预训练模型名称或路径")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="最大源序列长度")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="最大目标序列长度")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="Beam search 大小")
    
    # 训练参数
    parser.add_argument("--do_train", action='store_true',
                        help="是否训练")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否评估")
    parser.add_argument("--do_test", action='store_true',
                        help="是否测试")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="评估批次大小")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="训练轮数")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Warmup 步数")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="梯度累积步数")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    logger.info(f"Device: {args.device}, n_gpu: {args.n_gpu}")
    
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    
    model.to(args.device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info(f"Model loaded: {args.model_name_or_path}")
    
    # 训练
    if args.do_train:
        train_examples = read_examples(args.train_data_file)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_dataset = TextDataset(train_features, args)
        train(args, train_dataset, model, tokenizer)
    
    # 测试
    if args.do_test:
        checkpoint = os.path.join(args.output_dir, 'best_model', 'model.bin')
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint))
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
