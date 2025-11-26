#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缺陷检测评估器
评估指标: Accuracy, Precision, Recall, F1, AUC-ROC
"""

import sys
import argparse
import json
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


def main():
    parser = argparse.ArgumentParser(description='缺陷检测评估器')
    parser.add_argument('-a', '--answers', required=True, type=str,
                        help='真实答案文件路径 (每行一个 0/1)')
    parser.add_argument('-p', '--predictions', required=True, type=str,
                        help='预测结果文件路径 (每行一个 0/1)')
    parser.add_argument('--probs', type=str, default=None,
                        help='预测概率文件路径 (可选，用于计算 AUC)')
    
    args = parser.parse_args()
    
    # 读取数据
    with open(args.answers, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = [int(line.strip()) for line in f if line.strip()]
    
    probs = None
    if args.probs:
        with open(args.probs, 'r', encoding='utf-8') as f:
            probs = [float(line.strip()) for line in f if line.strip()]
    
    if len(predictions) != len(labels):
        print(f"Warning: predictions ({len(predictions)}) != labels ({len(labels)})")
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        if probs:
            probs = probs[:min_len]
    
    print("=" * 50)
    print("Defect Detection Evaluation Results")
    print("=" * 50)
    print(f"Number of samples: {len(predictions)}")
    print(f"Positive samples (has defect): {sum(labels)}")
    print(f"Negative samples (no defect): {len(labels) - sum(labels)}")
    print()
    
    # 基本指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    
    # AUC-ROC (需要概率)
    auc_roc = None
    avg_precision = None
    if probs and len(set(labels)) > 1:
        try:
            auc_roc = roc_auc_score(labels, probs)
            avg_precision = average_precision_score(labels, probs)
            print(f"AUC-ROC:      {auc_roc:.4f}")
            print(f"Avg Precision:{avg_precision:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
    
    print()
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(f"            Predicted")
    print(f"            0      1")
    print(f"Actual 0   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       1   {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print()
    print("=" * 50)
    
    # JSON 输出
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negatives': int(cm[0][0]),
        'false_positives': int(cm[0][1]),
        'false_negatives': int(cm[1][0]),
        'true_positives': int(cm[1][1]),
        'num_samples': len(predictions),
        'num_positive': sum(labels),
        'num_negative': len(labels) - sum(labels)
    }
    
    if auc_roc is not None:
        results['auc_roc'] = auc_roc
        results['average_precision'] = avg_precision
    
    print("\nJSON Results:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
