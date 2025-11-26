#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数据预处理脚本
从 Git_crawler1 爬取的 MR 数据中构造三个任务的数据集：
1. code-summarization: 代码摘要生成
3. code-completion: 代码补全
4. defect-detection: 缺陷检测
"""

import os
import json
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def parse_diff(diff_text: str) -> Tuple[List[str], List[str]]:
    """
    解析 diff 文本，返回删除行和新增行
    """
    old_lines = []
    new_lines = []
    
    for line in diff_text.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            old_lines.append(line[1:].strip())
        elif line.startswith('+') and not line.startswith('+++'):
            new_lines.append(line[1:].strip())
    
    return old_lines, new_lines


def extract_code_context(diff_text: str, context_lines: int = 5) -> str:
    """
    从 diff 中提取代码上下文（不变的行 + 变更行）
    """
    lines = diff_text.split('\n')
    context = []
    
    for line in lines:
        if line.startswith('@@'):
            continue
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            context.append(line[1:])
        elif line.startswith('+'):
            context.append(line[1:])
        elif line.startswith(' '):
            context.append(line[1:])
        else:
            context.append(line)
    
    return '\n'.join(context[:50])  # 限制长度


def is_code_file(file_path: str) -> bool:
    """
    判断是否是代码文件
    """
    code_extensions = {
        '.py', '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx',
        '.java', '.js', '.ts', '.go', '.rs', '.rb', '.sh', '.cmake'
    }
    ext = Path(file_path).suffix.lower()
    return ext in code_extensions


def is_bugfix_mr(mr_data: Dict) -> bool:
    """
    判断是否是 bug-fix 类型的 MR
    """
    title = mr_data.get('title', '').lower()
    description = mr_data.get('description', '') or ''
    description = description.lower()
    
    bugfix_keywords = [
        'fix', 'bug', 'issue', 'error', 'crash', 'fault', 'defect',
        'patch', 'hotfix', 'resolve', 'repair', 'correct', 'wrong',
        'problem', 'fail', 'broken', 'memory leak', 'segfault', 'nullptr'
    ]
    
    text = title + ' ' + description
    return any(keyword in text for keyword in bugfix_keywords)


def load_mr_data(data_dir: str) -> List[Dict]:
    """
    加载所有 MR 数据
    """
    mr_data_list = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return []
    
    for json_file in data_path.glob('mr_*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file'] = str(json_file)
                mr_data_list.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(mr_data_list)} MR files from {data_dir}")
    return mr_data_list


# ============================================================
# 任务1: 代码摘要生成
# ============================================================

def create_summarization_dataset(mr_data_list: List[Dict], output_dir: str, eval_only: bool = False):
    """
    创建代码摘要数据集
    输入: 代码 diff
    输出: MR title（作为摘要）
    """
    examples = []
    
    for idx, mr in enumerate(mr_data_list):
        title = mr.get('title', '').strip()
        changes = mr.get('changes', [])
        
        if not title or not changes:
            continue
        
        # 合并所有代码变更
        all_diffs = []
        for change in changes:
            if is_code_file(change.get('new_path', '') or change.get('old_path', '')):
                diff = change.get('diff', '')
                if diff:
                    all_diffs.append(diff)
        
        if not all_diffs:
            continue
        
        # 构造样本
        code_diff = '\n'.join(all_diffs)[:2000]  # 限制长度
        
        example = {
            'idx': idx,
            'code': code_diff,
            'summary': title,
            'mr_iid': mr.get('iid', ''),
        }
        examples.append(example)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(examples)
    
    if eval_only:
        # 只生成测试集
        output_file = os.path.join(output_dir, 'test.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Code Summarization - test: {len(examples)} examples -> {output_file}")
    else:
        # 划分 train/valid/test
        n = len(examples)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)
        
        train_data = examples[:train_end]
        valid_data = examples[train_end:valid_end]
        test_data = examples[valid_end:]
        
        for split, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            output_file = os.path.join(output_dir, f'{split}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"  Code Summarization - {split}: {len(data)} examples -> {output_file}")
    
    return len(examples)


# ============================================================
# 任务3: 代码补全
# ============================================================

def create_completion_dataset(mr_data_list: List[Dict], output_dir: str, eval_only: bool = False):
    """
    创建代码补全数据集
    从新增的代码行中，mask 掉部分内容作为补全目标
    """
    examples = []
    
    for idx, mr in enumerate(mr_data_list):
        changes = mr.get('changes', [])
        
        for change in changes:
            file_path = change.get('new_path', '') or change.get('old_path', '')
            if not is_code_file(file_path):
                continue
            
            diff = change.get('diff', '')
            if not diff:
                continue
            
            old_lines, new_lines = parse_diff(diff)
            
            # 使用新增的代码行构造补全样本
            for new_line in new_lines:
                if len(new_line) < 10:  # 跳过太短的行
                    continue
                
                # 找到可以切分的位置（运算符、括号等）
                split_points = []
                for i, char in enumerate(new_line):
                    if char in '=({[,:':
                        if i > 5 and i < len(new_line) - 5:
                            split_points.append(i + 1)
                
                if not split_points:
                    continue
                
                split_idx = random.choice(split_points)
                context = new_line[:split_idx].strip()
                target = new_line[split_idx:].strip()
                
                if len(context) < 5 or len(target) < 3:
                    continue
                
                example = {
                    'idx': len(examples),
                    'context': context,
                    'target': target,
                    'file_path': file_path,
                    'mr_iid': mr.get('iid', ''),
                }
                examples.append(example)
    
    # 限制样本数量
    if len(examples) > 50000:
        examples = random.sample(examples, 50000)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(examples)
    
    if eval_only:
        # 只生成测试集
        output_file = os.path.join(output_dir, 'test.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Code Completion - test: {len(examples)} examples -> {output_file}")
    else:
        # 划分 train/valid/test
        n = len(examples)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)
        
        train_data = examples[:train_end]
        valid_data = examples[train_end:valid_end]
        test_data = examples[valid_end:]
        
        for split, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            output_file = os.path.join(output_dir, f'{split}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"  Code Completion - {split}: {len(data)} examples -> {output_file}")
    
    return len(examples)


# ============================================================
# 任务4: 缺陷检测
# ============================================================

def create_defect_dataset(mr_data_list: List[Dict], output_dir: str, eval_only: bool = False):
    """
    创建缺陷检测数据集
    label=1: bug-fix MR 中修复前的代码（有缺陷）
    label=0: 非 bug-fix MR 的代码（假设无缺陷）
    """
    examples = []
    
    bugfix_mrs = [mr for mr in mr_data_list if is_bugfix_mr(mr)]
    normal_mrs = [mr for mr in mr_data_list if not is_bugfix_mr(mr)]
    
    print(f"  Found {len(bugfix_mrs)} bug-fix MRs, {len(normal_mrs)} normal MRs")
    
    # 从 bug-fix MR 中提取有缺陷的代码（修复前）
    for mr in bugfix_mrs:
        changes = mr.get('changes', [])
        
        for change in changes:
            file_path = change.get('old_path', '') or change.get('new_path', '')
            if not is_code_file(file_path):
                continue
            
            diff = change.get('diff', '')
            if not diff:
                continue
            
            old_lines, new_lines = parse_diff(diff)
            
            # 使用被删除的代码行（修复前的有缺陷代码）
            if old_lines:
                code = '\n'.join(old_lines)
                if len(code) > 50:
                    example = {
                        'idx': len(examples),
                        'func': code[:1000],  # 限制长度
                        'target': 1,  # 有缺陷
                        'mr_iid': mr.get('iid', ''),
                        'mr_title': mr.get('title', ''),
                    }
                    examples.append(example)
    
    # 从正常 MR 中提取无缺陷的代码
    bugfix_count = len([e for e in examples if e['target'] == 1])
    normal_count = 0
    
    for mr in normal_mrs:
        if normal_count >= bugfix_count * 2:  # 负样本不超过正样本的2倍
            break
        
        changes = mr.get('changes', [])
        
        for change in changes:
            if normal_count >= bugfix_count * 2:
                break
            
            file_path = change.get('new_path', '') or change.get('old_path', '')
            if not is_code_file(file_path):
                continue
            
            diff = change.get('diff', '')
            if not diff:
                continue
            
            # 使用新增的代码（假设无缺陷）
            _, new_lines = parse_diff(diff)
            
            if new_lines:
                code = '\n'.join(new_lines)
                if len(code) > 50:
                    example = {
                        'idx': len(examples),
                        'func': code[:1000],
                        'target': 0,  # 无缺陷
                        'mr_iid': mr.get('iid', ''),
                        'mr_title': mr.get('title', ''),
                    }
                    examples.append(example)
                    normal_count += 1
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(examples)
    
    if eval_only:
        # 只生成测试集
        output_file = os.path.join(output_dir, 'test.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Defect Detection - test: {len(examples)} examples -> {output_file}")
        
        # 统计标签分布
        pos = sum(1 for e in examples if e['target'] == 1)
        neg = len(examples) - pos
        print(f"    test - positive: {pos}, negative: {neg}")
    else:
        # 划分 train/valid/test
        n = len(examples)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)
        
        train_data = examples[:train_end]
        valid_data = examples[train_end:valid_end]
        test_data = examples[valid_end:]
        
        for split, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            output_file = os.path.join(output_dir, f'{split}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"  Defect Detection - {split}: {len(data)} examples -> {output_file}")
        
        # 统计标签分布
        for split, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            pos = sum(1 for e in data if e['target'] == 1)
            neg = len(data) - pos
            print(f"    {split} - positive: {pos}, negative: {neg}")
    
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description='从 MR 数据构造评估任务数据集')
    parser.add_argument('--mr_data_dir', type=str, required=True,
                        help='MR 数据目录 (包含 mr_*.json 文件)')
    parser.add_argument('--output_base_dir', type=str, default='.',
                        help='输出基础目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--tasks', type=str, default='all',
                        help='要构造的任务: all, summarization, completion, defect')
    parser.add_argument('--eval_only', action='store_true',
                        help='只生成测试集 (用于评估已训练的模型)')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print("=" * 60)
    print("粒子物理代码评估 - 数据预处理")
    print("=" * 60)
    
    # 加载 MR 数据
    mr_data_list = load_mr_data(args.mr_data_dir)
    
    if not mr_data_list:
        print("Error: No MR data found!")
        return
    
    print(f"\nTotal MRs loaded: {len(mr_data_list)}")
    if args.eval_only:
        print("Mode: Evaluation only (只生成测试集)")
    print()
    
    tasks = args.tasks.lower()
    
    # 任务1: 代码摘要生成
    if tasks in ['all', 'summarization']:
        print("Creating Code Summarization dataset...")
        output_dir = os.path.join(args.output_base_dir, 'code-summarization', 'dataset')
        count = create_summarization_dataset(mr_data_list, output_dir, args.eval_only)
        print(f"  Total: {count} examples\n")
    
    # 任务3: 代码补全
    if tasks in ['all', 'completion']:
        print("Creating Code Completion dataset...")
        output_dir = os.path.join(args.output_base_dir, 'code-completion', 'dataset')
        count = create_completion_dataset(mr_data_list, output_dir, args.eval_only)
        print(f"  Total: {count} examples\n")
    
    # 任务4: 缺陷检测
    if tasks in ['all', 'defect']:
        print("Creating Defect Detection dataset...")
        output_dir = os.path.join(args.output_base_dir, 'defect-detection', 'dataset')
        count = create_defect_dataset(mr_data_list, output_dir, args.eval_only)
        print(f"  Total: {count} examples\n")
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
