#!/usr/bin/env python3
"""
MR 数据预处理脚本 - 将爬取的 GitLab MR 数据转换为 CodeXGLUE 格式

用途：从 Git_crawler1 爬取的 mr_*.json 文件创建代码审查意图分类数据集
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import argparse


class MRDataProcessor:
    """处理 MR 数据并转换为训练格式"""
    
    def __init__(self, mr_data_dir: str, output_dir: str):
        self.mr_data_dir = Path(mr_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = defaultdict(int)
        
    def extract_code_from_changes(self, changes: List[Dict]) -> str:
        """从 MR changes 中提取代码片段"""
        code_snippets = []
        
        for change in changes:
            new_path = change.get('new_path', '')
            diff = change.get('diff', '')
            
            # 过滤：只保留 C++, Python, ROOT macro 文件
            physics_extensions = ['.cpp', '.cxx', '.cc', '.py', '.C', '.h', '.hpp', '.hh', '.cu']
            if not any(new_path.endswith(ext) for ext in physics_extensions):
                continue
            
            # 提取新增/修改的代码行
            new_lines = []
            for line in diff.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    new_lines.append(line[1:].strip())
            
            if new_lines:
                code_snippets.append({
                    'file': new_path,
                    'code': '\n'.join(new_lines[:50])  # 限制长度
                })
        
        # 合并所有代码片段
        combined_code = '\n'.join([
            f"// File: {s['file']}\n{s['code']}" 
            for s in code_snippets[:3]  # 最多3个文件
        ])
        
        return combined_code
    
    def classify_review_intent(self, comment: str) -> int:
        """
        分类审查意图
        0 - 建议优化 (suggest optimization)
        1 - 指出错误 (point out bug/error)
        2 - 请求澄清 (request clarification)
        3 - 批准通过 (approve/LGTM)
        """
        comment_lower = comment.lower()
        
        # 错误相关关键词
        error_keywords = ['bug', 'error', 'wrong', 'incorrect', 'leak', 'crash', 
                         'segfault', 'fix', 'broken', 'fail', 'issue']
        if any(kw in comment_lower for kw in error_keywords):
            self.stats['intent_error'] += 1
            return 1
        
        # 批准相关关键词
        approve_keywords = ['lgtm', 'looks good', 'approve', 'approved', 
                           '👍', '+1', 'merge']
        if any(kw in comment_lower for kw in approve_keywords):
            self.stats['intent_approve'] += 1
            return 3
        
        # 澄清相关关键词
        clarify_keywords = ['why', 'how', 'clarify', 'explain', 'what', 
                           'could you', 'can you', '?']
        if any(kw in comment_lower for kw in clarify_keywords):
            self.stats['intent_clarify'] += 1
            return 2
        
        # 默认为优化建议
        self.stats['intent_optimize'] += 1
        return 0
    
    def is_physics_related(self, text: str) -> bool:
        """判断是否与粒子物理相关"""
        physics_keywords = [
            # ROOT 相关
            'TTree', 'TBranch', 'TH1', 'TH2', 'TCanvas', 'ROOT',
            # Geant4 相关
            'G4', 'Geant4', 'G4Step', 'G4Track', 'G4Event',
            # 物理概念
            'particle', 'detector', 'energy', 'momentum', 'GeV', 'TeV',
            'Monte Carlo', 'simulation', 'reconstruction', 'trigger',
            # 常见库
            'CMSSW', 'Athena', 'GaudiKernel', 'ALICE', 'ATLAS', 'CMS'
        ]
        
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in physics_keywords)
    
    def process_mr_file(self, mr_file: Path) -> List[Dict[str, Any]]:
        """处理单个 MR 文件，返回数据样本列表"""
        with open(mr_file, 'r', encoding='utf-8') as f:
            mr = json.load(f)
        
        samples = []
        
        # 提取代码变更
        code_changes = self.extract_code_from_changes(mr.get('changes', []))
        
        if not code_changes:
            self.stats['skipped_no_code'] += 1
            return samples
        
        # 提取 MR 上下文
        mr_context = f"Title: {mr.get('title', '')}\nDescription: {mr.get('description', '')}"
        
        # 处理每条非系统评论
        comments = mr.get('comments', [])
        for comment in comments:
            if comment.get('system', False):
                continue
            
            comment_body = comment.get('body', '').strip()
            if len(comment_body) < 10:  # 过滤太短的评论
                continue
            
            # 分类意图
            intent = self.classify_review_intent(comment_body)
            
            # 构建样本
            sample = {
                'idx': len(samples),
                'code': code_changes,
                'context': mr_context,
                'comment': comment_body,
                'target': intent,
                'mr_iid': mr.get('iid'),
                'mr_url': mr.get('web_url', '')
            }
            
            samples.append(sample)
            self.stats['total_samples'] += 1
        
        return samples
    
    def process_all_mrs(self) -> List[Dict[str, Any]]:
        """处理所有 MR 文件"""
        all_samples = []
        mr_files = list(self.mr_data_dir.glob('mr_*.json'))
        
        print(f"找到 {len(mr_files)} 个 MR 文件")
        
        for i, mr_file in enumerate(mr_files, 1):
            if i % 10 == 0:
                print(f"处理进度: {i}/{len(mr_files)}")
            
            try:
                samples = self.process_mr_file(mr_file)
                all_samples.extend(samples)
            except Exception as e:
                print(f"处理 {mr_file.name} 时出错: {e}")
                self.stats['errors'] += 1
        
        return all_samples
    
    def split_dataset(self, samples: List[Dict], train_ratio=0.8, valid_ratio=0.1):
        """划分训练集、验证集、测试集"""
        import random
        random.seed(42)
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        train_samples = samples[:train_end]
        valid_samples = samples[train_end:valid_end]
        test_samples = samples[valid_end:]
        
        return train_samples, valid_samples, test_samples
    
    def save_as_jsonl(self, samples: List[Dict], filename: str):
        """保存为 JSONL 格式"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"已保存: {filepath} ({len(samples)} 样本)")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("数据处理统计:")
        print("="*50)
        for key, value in sorted(self.stats.items()):
            print(f"{key:.<30} {value}")
        print("="*50)
    
    def run(self):
        """执行完整的数据处理流程"""
        print("开始处理 MR 数据...")
        
        # 处理所有 MR
        all_samples = self.process_all_mrs()
        
        if len(all_samples) == 0:
            print("❌ 没有生成任何样本，请检查数据源")
            return
        
        # 划分数据集
        train, valid, test = self.split_dataset(all_samples)
        
        # 保存数据集
        self.save_as_jsonl(train, 'train.jsonl')
        self.save_as_jsonl(valid, 'valid.jsonl')
        self.save_as_jsonl(test, 'test.jsonl')
        
        # 打印统计
        self.print_statistics()
        
        # 保存类别分布
        self.save_label_distribution(all_samples)
    
    def save_label_distribution(self, samples: List[Dict]):
        """保存标签分布统计"""
        label_dist = defaultdict(int)
        for s in samples:
            label_dist[s['target']] += 1
        
        label_names = {
            0: 'Optimization Suggestion',
            1: 'Bug/Error Report',
            2: 'Clarification Request',
            3: 'Approval/LGTM'
        }
        
        print("\n标签分布:")
        for label, count in sorted(label_dist.items()):
            print(f"  {label} ({label_names[label]}): {count} ({count/len(samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='将 GitLab MR 数据转换为代码审查意图分类数据集'
    )
    parser.add_argument(
        '--mr_data_dir',
        default='../../../../Git_crawler1/mr_data',
        help='MR 数据目录（包含 mr_*.json 文件）'
    )
    parser.add_argument(
        '--output_dir',
        default='.',
        help='输出目录'
    )
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.mr_data_dir).exists():
        print(f"❌ MR 数据目录不存在: {args.mr_data_dir}")
        print(f"提示：请先使用 Git_crawler1 爬取 MR 数据")
        print(f"示例：cd Git_crawler1 && python crawler.py --project-id your-project")
        return
    
    # 执行处理
    processor = MRDataProcessor(args.mr_data_dir, args.output_dir)
    processor.run()
    
    print("\n✅ 数据处理完成！")
    print(f"下一步：cd ../code && python run.py --do_train")


if __name__ == '__main__':
    main()
