#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码摘要生成评估器
评估指标: BLEU, ROUGE, METEOR
"""

import sys
import argparse
import math
from collections import Counter


def get_ngrams(tokens, n):
    """获取 n-gram"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(predictions, references, max_n=4):
    """计算语料级别 BLEU"""
    
    total_matches = [0] * max_n
    total_counts = [0] * max_n
    total_ref_len = 0
    total_pred_len = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        total_ref_len += len(ref_tokens)
        total_pred_len += len(pred_tokens)
        
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            total_matches[n-1] += overlap
            total_counts[n-1] += total
    
    # 计算精度
    precisions = []
    for n in range(max_n):
        if total_counts[n] > 0:
            precisions.append(total_matches[n] / total_counts[n])
        else:
            precisions.append(0.0)
    
    # 几何平均
    if min(precisions) > 0:
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        geo_mean = math.exp(log_avg)
    else:
        geo_mean = 0.0
    
    # 简短惩罚
    if total_pred_len > 0:
        bp = min(1.0, math.exp(1 - total_ref_len / total_pred_len))
    else:
        bp = 0.0
    
    return bp * geo_mean * 100


def compute_rouge_l(predictions, references):
    """计算 ROUGE-L (基于最长公共子序列)"""
    
    def lcs_length(x, y):
        """最长公共子序列长度"""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        if len(pred_tokens) > 0:
            precision = lcs_len / len(pred_tokens)
        else:
            precision = 0.0
        
        if len(ref_tokens) > 0:
            recall = lcs_len / len(ref_tokens)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(predictions)
    return {
        'precision': total_precision / n * 100 if n > 0 else 0,
        'recall': total_recall / n * 100 if n > 0 else 0,
        'f1': total_f1 / n * 100 if n > 0 else 0,
    }


def compute_exact_match(predictions, references):
    """计算精确匹配率"""
    matches = sum(1 for p, r in zip(predictions, references) 
                  if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) * 100 if predictions else 0


def main():
    parser = argparse.ArgumentParser(description='代码摘要生成评估器')
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
    print("Code Summarization Evaluation Results")
    print("=" * 50)
    print(f"Number of samples: {len(predictions)}")
    print()
    
    # BLEU
    bleu = compute_bleu(predictions, references)
    print(f"BLEU-4:         {bleu:.2f}")
    
    # ROUGE-L
    rouge = compute_rouge_l(predictions, references)
    print(f"ROUGE-L F1:     {rouge['f1']:.2f}")
    print(f"ROUGE-L P:      {rouge['precision']:.2f}")
    print(f"ROUGE-L R:      {rouge['recall']:.2f}")
    
    # Exact Match
    em = compute_exact_match(predictions, references)
    print(f"Exact Match:    {em:.2f}")
    
    print()
    print("=" * 50)
    
    # 输出 JSON 格式结果
    results = {
        'bleu': bleu,
        'rouge_l_f1': rouge['f1'],
        'rouge_l_precision': rouge['precision'],
        'rouge_l_recall': rouge['recall'],
        'exact_match': em,
        'num_samples': len(predictions)
    }
    
    import json
    print("\nJSON Results:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
