#!/usr/bin/env python3
"""
MR 数据分析脚本 - 分析爬取的数据，确定可行的评估任务

用法:
    python analyze_mr_data.py --data_dir=../Git_crawler1/mr_data
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import re


def load_mr_data(data_dir: str):
    """加载所有 MR 数据"""
    mr_files = list(Path(data_dir).glob('mr_*.json'))
    mrs = []
    
    for f in mr_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                mrs.append(json.load(fp))
        except Exception as e:
            print(f"  ⚠️ 无法加载 {f.name}: {e}")
    
    return mrs


def analyze_basic_stats(mrs):
    """基础统计"""
    print("\n" + "=" * 60)
    print("📊 基础统计")
    print("=" * 60)
    
    print(f"总 MR 数量: {len(mrs)}")
    
    # 状态统计
    states = Counter(mr.get('state', 'unknown') for mr in mrs)
    print(f"MR 状态分布: {dict(states)}")
    
    # 时间范围
    dates = [mr.get('created_at', '')[:10] for mr in mrs if mr.get('created_at')]
    if dates:
        print(f"时间范围: {min(dates)} ~ {max(dates)}")


def analyze_code_changes(mrs):
    """分析代码变更"""
    print("\n" + "=" * 60)
    print("📝 代码变更分析")
    print("=" * 60)
    
    total_changes = 0
    file_extensions = Counter()
    diff_lengths = []
    
    for mr in mrs:
        changes = mr.get('changes', [])
        total_changes += len(changes)
        
        for change in changes:
            # 文件扩展名
            path = change.get('new_path', '') or change.get('old_path', '')
            if '.' in path:
                ext = '.' + path.rsplit('.', 1)[-1]
                file_extensions[ext] += 1
            
            # Diff 长度
            diff = change.get('diff', '')
            diff_lengths.append(len(diff))
    
    print(f"总变更文件数: {total_changes}")
    print(f"平均每个 MR 变更文件数: {total_changes / len(mrs):.1f}")
    
    print("\n文件类型分布 (Top 10):")
    for ext, count in file_extensions.most_common(10):
        print(f"  {ext}: {count}")
    
    if diff_lengths:
        print(f"\nDiff 长度: min={min(diff_lengths)}, max={max(diff_lengths)}, avg={sum(diff_lengths)/len(diff_lengths):.0f}")
    
    # 物理相关文件
    physics_exts = ['.cpp', '.cxx', '.cc', '.C', '.h', '.hpp', '.py', '.cu']
    physics_count = sum(file_extensions.get(ext, 0) for ext in physics_exts)
    print(f"\n物理相关文件 (C++/Python): {physics_count} ({physics_count/total_changes*100:.1f}%)")
    
    return file_extensions


def analyze_comments(mrs):
    """分析评论数据"""
    print("\n" + "=" * 60)
    print("💬 评论分析")
    print("=" * 60)
    
    total_comments = 0
    non_system_comments = 0
    comment_lengths = []
    
    for mr in mrs:
        comments = mr.get('comments', [])
        total_comments += len(comments)
        
        for comment in comments:
            if not comment.get('system', False):
                non_system_comments += 1
                body = comment.get('body', '')
                comment_lengths.append(len(body))
    
    print(f"总评论数: {total_comments}")
    print(f"非系统评论数: {non_system_comments}")
    print(f"平均每个 MR 评论数: {total_comments / len(mrs):.1f}")
    
    if comment_lengths:
        print(f"评论长度: min={min(comment_lengths)}, max={max(comment_lengths)}, avg={sum(comment_lengths)/len(comment_lengths):.0f}")
        
        # 有效评论（长度 > 10）
        valid_comments = sum(1 for l in comment_lengths if l > 10)
        print(f"有效评论数 (长度>10): {valid_comments}")
    
    return non_system_comments, comment_lengths


def analyze_discussions(mrs):
    """分析讨论数据"""
    print("\n" + "=" * 60)
    print("🗣️ 讨论分析")
    print("=" * 60)
    
    total_discussions = 0
    multi_turn = 0  # 多轮对话
    qa_pairs = []   # 问答对
    
    for mr in mrs:
        discussions = mr.get('discussions', [])
        total_discussions += len(discussions)
        
        for disc in discussions:
            notes = disc.get('notes', [])
            if len(notes) > 1:
                multi_turn += 1
                # 提取问答对
                first_note = notes[0].get('body', '')
                if '?' in first_note and len(notes) > 1:
                    qa_pairs.append({
                        'question': first_note,
                        'answer': notes[1].get('body', '')
                    })
    
    print(f"总讨论数: {total_discussions}")
    print(f"多轮对话数: {multi_turn}")
    print(f"潜在问答对数: {len(qa_pairs)}")
    
    return qa_pairs


def analyze_titles_descriptions(mrs):
    """分析标题和描述"""
    print("\n" + "=" * 60)
    print("📋 标题/描述分析")
    print("=" * 60)
    
    title_lengths = [len(mr.get('title', '')) for mr in mrs]
    desc_lengths = [len(mr.get('description', '') or '') for mr in mrs]
    
    non_empty_desc = sum(1 for d in desc_lengths if d > 0)
    
    print(f"标题长度: min={min(title_lengths)}, max={max(title_lengths)}, avg={sum(title_lengths)/len(title_lengths):.0f}")
    print(f"有描述的 MR: {non_empty_desc} ({non_empty_desc/len(mrs)*100:.1f}%)")
    
    if non_empty_desc > 0:
        valid_desc = [d for d in desc_lengths if d > 0]
        print(f"描述长度: min={min(valid_desc)}, max={max(valid_desc)}, avg={sum(valid_desc)/len(valid_desc):.0f}")


def suggest_tasks(mrs, file_extensions, comment_count, qa_pairs):
    """根据数据量建议可行的任务"""
    print("\n" + "=" * 60)
    print("🎯 可行任务建议")
    print("=" * 60)
    
    n = len(mrs)
    
    print(f"\n基于 {n} 个 MR 数据:\n")
    
    # 任务1: 代码摘要
    if n >= 50:
        print("✅ 任务1: 代码变更摘要生成")
        print(f"   数据量: {n} 个样本 (每个MR一个)")
        print("   推荐: Few-shot 或 Fine-tune (如果>500)")
    else:
        print("⚠️ 任务1: 代码变更摘要生成 - 数据量不足")
    
    # 任务2: 评论生成
    if comment_count >= 100:
        print(f"\n✅ 任务2: 代码审查评论生成")
        print(f"   数据量: {comment_count} 条评论")
    else:
        print(f"\n⚠️ 任务2: 代码审查评论生成 - 评论数量不足 ({comment_count})")
    
    # 任务3: 代码补全
    physics_exts = ['.cpp', '.cxx', '.cc', '.C', '.h', '.hpp', '.py']
    physics_files = sum(file_extensions.get(ext, 0) for ext in physics_exts)
    if physics_files >= 100:
        print(f"\n✅ 任务3: 物理代码补全")
        print(f"   数据量: {physics_files} 个物理相关文件")
    else:
        print(f"\n⚠️ 任务3: 物理代码补全 - 物理文件不足 ({physics_files})")
    
    # 任务4: 代码问答
    if len(qa_pairs) >= 50:
        print(f"\n✅ 任务4: 代码理解问答")
        print(f"   数据量: {len(qa_pairs)} 个问答对")
    else:
        print(f"\n⚠️ 任务4: 代码理解问答 - 问答对不足 ({len(qa_pairs)})")
    
    # 任务5: 意图分类
    if comment_count >= 200:
        print(f"\n✅ 任务5: 评论意图分类")
        print(f"   数据量: {comment_count} 条评论")
    else:
        print(f"\n⚠️ 任务5: 评论意图分类 - 评论不足")
    
    print("\n" + "-" * 60)
    print("建议优先级:")
    if n >= 100:
        print("1. 先做代码摘要生成（数据充足）")
        print("2. 再做意图分类（如果评论足够）")
        print("3. 最后尝试代码补全")
    else:
        print("1. 数据量较少，建议使用 few-shot 评估")
        print("2. 或继续爬取更多数据")


def save_sample_data(mrs, output_dir):
    """保存样本数据供检查"""
    sample_file = Path(output_dir) / 'sample_data.json'
    
    if len(mrs) > 0:
        sample = mrs[0]
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"\n📄 样本数据已保存到: {sample_file}")


def main():
    parser = argparse.ArgumentParser(description='分析 MR 数据')
    parser.add_argument('--data_dir', required=True, help='MR 数据目录')
    parser.add_argument('--output_dir', default='.', help='输出目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 MR 数据分析工具")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    
    # 检查目录
    if not Path(args.data_dir).exists():
        print(f"❌ 目录不存在: {args.data_dir}")
        print("请先使用 Git_crawler1 爬取数据")
        return
    
    # 加载数据
    print("\n加载数据...")
    mrs = load_mr_data(args.data_dir)
    
    if len(mrs) == 0:
        print("❌ 没有找到任何 MR 数据")
        return
    
    # 运行分析
    analyze_basic_stats(mrs)
    file_extensions = analyze_code_changes(mrs)
    comment_count, _ = analyze_comments(mrs)
    qa_pairs = analyze_discussions(mrs)
    analyze_titles_descriptions(mrs)
    
    # 建议任务
    suggest_tasks(mrs, file_extensions, comment_count, qa_pairs)
    
    # 保存样本
    save_sample_data(mrs, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
