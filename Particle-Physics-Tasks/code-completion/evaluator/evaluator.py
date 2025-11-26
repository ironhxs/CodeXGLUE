#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码补全评估器
评估指标: Edit Similarity, Exact Match, BLEU
"""

import sys
import argparse
import json
from fuzzywuzzy import fuzz
import math
from collections import Counter


def compute_edit_similarity(predictions, references):
    """计算编辑距离相似度"""
    similarities = []
    for pred, ref in zip(predictions, references):
        sim = fuzz.ratio(pred.strip(), ref.strip())
        similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0


def compute_exact_match(predictions, references):
    """计算精确匹配率"""
    matches = sum(1 for p, r in zip(predictions, references) 
                  if p.strip() == r.strip())
    return matches / len(predictions) * 100 if predictions else 0


def get_ngrams(tokens, n):
    """获取 n-gram"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(predictions, references, max_n=4):
    """计算 BLEU"""
    total_matches = [0] * max_n
    total_counts = [0] * max_n
    total_ref_len = 0
    total_pred_len = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred.strip())  # 字符级别
        ref_tokens = list(ref.strip())
        
        total_ref_len += len(ref_tokens)
        total_pred_len += len(pred_tokens)
        
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            total_matches[n-1] += overlap
            total_counts[n-1] += total
    
    precisions = []
    for n in range(max_n):
        if total_counts[n] > 0:
            precisions.append(total_matches[n] / total_counts[n])
        else:
            precisions.append(0.0)
    
    if min(precisions) > 0:
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        geo_mean = math.exp(log_avg)
    else:
        geo_mean = 0.0
    
    if total_pred_len > 0:
        bp = min(1.0, math.exp(1 - total_ref_len / total_pred_len))
    else:
        bp = 0.0
    
    return bp * geo_mean * 100


def compute_levenshtein(s1, s2):
    """计算 Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_normalized_edit_distance(predictions, references):
    """计算归一化编辑距离 (越低越好，转换为越高越好)"""
    distances = []
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        dist = compute_levenshtein(pred, ref)
        max_len = max(len(pred), len(ref), 1)
        normalized = 1 - dist / max_len  # 转换为相似度
        distances.append(normalized * 100)
    return sum(distances) / len(distances) if distances else 0


def main():
    parser = argparse.ArgumentParser(description='代码补全评估器')
    parser.add_argument('-a', '--answers', required=True, type=str,
                        help='真实答案文件路径 (每行一个)')
    parser.add_argument('-p', '--predictions', required=True, type=str,
                        help='预测结果文件路径 (每行一个)')
    
    args = parser.parse_args()
    
    # 读取数据
    with open(args.answers, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    if len(predictions) != len(references):
        print(f"Warning: predictions ({len(predictions)}) != references ({len(references)})")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    print("=" * 50)
    print("Code Completion Evaluation Results")
    print("=" * 50)
    print(f"Number of samples: {len(predictions)}")
    print()
    
    # Edit Similarity (fuzzywuzzy)
    edit_sim = compute_edit_similarity(predictions, references)
    print(f"Edit Similarity:      {edit_sim:.2f}")
    
    # Normalized Edit Distance
    ned = compute_normalized_edit_distance(predictions, references)
    print(f"Norm Edit Distance:   {ned:.2f}")
    
    # Exact Match
    em = compute_exact_match(predictions, references)
    print(f"Exact Match:          {em:.2f}")
    
    # BLEU (字符级别)
    bleu = compute_bleu(predictions, references)
    print(f"Char-BLEU:            {bleu:.2f}")
    
    print()
    print("=" * 50)
    
    # JSON 输出
    results = {
        'edit_similarity': edit_sim,
        'normalized_edit_distance': ned,
        'exact_match': em,
        'char_bleu': bleu,
        'num_samples': len(predictions)
    }
    
    print("\nJSON Results:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
