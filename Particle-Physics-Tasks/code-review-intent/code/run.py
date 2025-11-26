# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
代码审查意图分类 - 训练和评估脚本

用法:
    python run.py \\
        --model_name_or_path=microsoft/codebert-base \\
        --do_train --do_eval --do_test \\
        --train_data_file=../dataset/train.jsonl \\
        --eval_data_file=../dataset/valid.jsonl \\
        --test_data_file=../dataset/test.jsonl \\
        --output_dir=../saved_models \\
        --num_labels=4 \\
        --block_size=512 \\
        --train_batch_size=16 \\
        --eval_batch_size=32 \\
        --learning_rate=2e-5 \\
        --num_train_epochs=5
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    RobertaConfig, 
    RobertaModel, 
    RobertaTokenizer
)

from model import Model

logger = logging.getLogger(__name__)

# 模型类映射
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


class InputFeatures(object):
    """单个训练/测试样本的特征"""
    def __init__(self, input_ids, label, idx):
        self.input_ids = input_ids
        self.label = label
        self.idx = idx


class TextDataset(Dataset):
    """代码审查意图分类数据集"""
    
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # 组合代码、上下文和评论
                code = data.get('code', '')
                context = data.get('context', '')
                comment = data.get('comment', '')
                
                # 拼接输入文本
                input_text = f"{context}\n\nCode Changes:\n{code}\n\nReview Comment:\n{comment}"
                
                # Tokenize
                input_ids = tokenizer.encode(
                    input_text,
                    max_length=args.block_size,
                    padding='max_length',
                    truncation=True
                )
                
                # 获取标签
                label = int(data.get('target', 0))
                idx = data.get('idx', 0)
                
                self.examples.append(
                    InputFeatures(input_ids, label, idx)
                )
        
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label)
        )


def set_seed(args):
    """设置随机种子"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """训练模型"""
    
    # 准备数据加载器
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size
    )
    
    # 准备优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # 计算总步数
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max_steps * 0.1,
        num_training_steps=max_steps
    )
    
    # 训练循环
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    model.train()
    model.zero_grad()
    
    best_acc = 0
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            
            loss, _ = model(input_ids, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 评估
        if args.do_eval:
            results = evaluate(args, model, tokenizer)
            logger.info(f"Epoch {epoch+1} - Eval Accuracy: {results['eval_acc']:.4f}")
            
            # 保存最佳模型
            if results['eval_acc'] > best_acc:
                best_acc = results['eval_acc']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                os.makedirs(output_dir, exist_ok=True)
                
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                tokenizer.save_pretrained(output_dir)
                
                logger.info(f"Saving best model to {output_dir}")


def evaluate(args, model, tokenizer):
    """评估模型"""
    
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        
        with torch.no_grad():
            loss, probs = model(input_ids, label)
            eval_loss += loss.item()
        
        nb_eval_steps += 1
        preds.append(probs.cpu().numpy())
        labels.append(label.cpu().numpy())
    
    # 计算指标
    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    preds = np.argmax(preds, axis=1)
    
    acc = np.mean(preds == labels)
    
    result = {
        'eval_loss': eval_loss / nb_eval_steps,
        'eval_acc': acc
    }
    
    return result


def test(args, model, tokenizer):
    """测试模型"""
    
    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    
    model.eval()
    preds = []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids = batch[0].to(args.device)
        
        with torch.no_grad():
            probs = model(input_ids)
        
        preds.append(probs.cpu().numpy())
    
    # 保存预测结果
    preds = np.concatenate(preds, 0)
    preds = np.argmax(preds, axis=1)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "predictions.txt"), 'w') as f:
        for pred in preds:
            f.write(str(pred) + '\n')
    
    logger.info(f"Predictions saved to {os.path.join(output_dir, 'predictions.txt')}")


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--train_data_file", default=None, type=str, required=True)
    parser.add_argument("--eval_data_file", default=None, type=str, required=True)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    
    # 模型参数
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--num_labels", default=4, type=int, help="Number of intent classes")
    
    # 训练参数
    parser.add_argument("--block_size", default=512, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    
    logger.info(f"Device: {args.device}, n_gpu: {args.n_gpu}")
    
    # 设置随机种子
    set_seed(args)
    
    # 加载模型
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    config.num_labels = args.num_labels
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    )
    
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = Model(encoder, config, tokenizer, args)
    
    model.to(args.device)
    
    logger.info(f"Training/evaluation parameters {args}")
    
    # 训练
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)
    
    # 测试
    if args.do_test:
        # 加载最佳模型
        checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            logger.info(f"Loaded best model from {checkpoint}")
        
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
