#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
评估器 - 计算代码审查意图分类的性能指标

用法:
    python evaluator.py -a ../dataset/test.jsonl -p ../saved_models/predictions.txt
"""

import argparse
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def read_answers(filename):
    """读取真实标签"""
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(int(data['target']))
    return labels


def read_predictions(filename):
    """读取预测结果"""
    predictions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(int(line.strip()))
    return predictions


def calculate_scores(answers, predictions):
    """计算分类指标"""
    
    # 整体准确率
    accuracy = accuracy_score(answers, predictions)
    
    # 每个类别的 Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        answers, 
        predictions, 
        average=None,
        labels=[0, 1, 2, 3]
    )
    
    # Macro 平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        answers, 
        predictions, 
        average='macro'
    )
    
    # 类别名称
    label_names = {
        0: 'Optimization Suggestion',
        1: 'Bug/Error Report',
        2: 'Clarification Request',
        3: 'Approval/LGTM'
    }
    
    # 构建结果字典
    results = {
        'Accuracy': round(accuracy, 4),
        'Macro Precision': round(macro_precision, 4),
        'Macro Recall': round(macro_recall, 4),
        'Macro F1': round(macro_f1, 4),
        'Per-class Metrics': {}
    }
    
    for i in range(4):
        results['Per-class Metrics'][f'Class {i} ({label_names[i]})'] = {
            'Precision': round(precision[i], 4),
            'Recall': round(recall[i], 4),
            'F1': round(f1[i], 4),
            'Support': int(support[i])
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估代码审查意图分类结果')
    parser.add_argument('-a', '--answers', required=True, help='真实标签文件 (JSONL)')
    parser.add_argument('-p', '--predictions', required=True, help='预测结果文件 (TXT)')
    args = parser.parse_args()
    
    # 读取数据
    print("读取真实标签...")
    answers = read_answers(args.answers)
    
    print("读取预测结果...")
    predictions = read_predictions(args.predictions)
    
    # 检查长度
    if len(answers) != len(predictions):
        print(f"❌ 错误: 标签数量 ({len(answers)}) 与预测数量 ({len(predictions)}) 不匹配")
        return
    
    print(f"共 {len(answers)} 个样本\n")
    
    # 计算指标
    results = calculate_scores(answers, predictions)
    
    # 打印结果
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"Accuracy:        {results['Accuracy']:.4f}")
    print(f"Macro Precision: {results['Macro Precision']:.4f}")
    print(f"Macro Recall:    {results['Macro Recall']:.4f}")
    print(f"Macro F1:        {results['Macro F1']:.4f}")
    print()
    print("Per-class Metrics:")
    print("-" * 60)
    
    for class_name, metrics in results['Per-class Metrics'].items():
        print(f"\n{class_name}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  Support:   {metrics['Support']}")
    
    print("=" * 60)
    
    # 保存结果为 JSON
    import os
    output_dir = os.path.dirname(args.predictions)
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


if __name__ == '__main__':
    main()
