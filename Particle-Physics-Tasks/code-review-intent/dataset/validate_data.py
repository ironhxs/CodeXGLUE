#!/usr/bin/env python3
"""
验证数据集格式是否正确
"""

import json
import sys
from pathlib import Path


def validate_jsonl_file(filepath):
    """验证 JSONL 文件格式"""
    print(f"\n检查文件: {filepath}")
    
    if not Path(filepath).exists():
        print(f"  ❌ 文件不存在")
        return False
    
    required_fields = ['idx', 'code', 'context', 'comment', 'target']
    samples = []
    errors = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 检查必需字段
                missing = [field for field in required_fields if field not in data]
                if missing:
                    errors.append(f"  行 {i}: 缺少字段 {missing}")
                    continue
                
                # 检查标签范围
                if not 0 <= data['target'] <= 3:
                    errors.append(f"  行 {i}: 标签值 {data['target']} 超出范围 [0-3]")
                
                samples.append(data)
                
            except json.JSONDecodeError as e:
                errors.append(f"  行 {i}: JSON 解析错误 - {e}")
    
    # 打印统计
    print(f"  ✅ 总样本数: {len(samples)}")
    
    if samples:
        # 标签分布
        label_dist = {}
        for s in samples:
            label = s['target']
            label_dist[label] = label_dist.get(label, 0) + 1
        
        print(f"  📊 标签分布:")
        label_names = {
            0: '优化建议',
            1: '错误报告',
            2: '澄清请求',
            3: '批准通过'
        }
        for label in sorted(label_dist.keys()):
            count = label_dist[label]
            pct = count / len(samples) * 100
            print(f"     {label} ({label_names.get(label, '未知')}): {count} ({pct:.1f}%)")
        
        # 代码长度统计
        code_lengths = [len(s['code']) for s in samples]
        print(f"  📏 代码长度: min={min(code_lengths)}, max={max(code_lengths)}, avg={sum(code_lengths)/len(code_lengths):.0f}")
    
    # 打印错误
    if errors:
        print(f"\n  ⚠️  发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个
            print(error)
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
        return False
    
    return True


def main():
    dataset_dir = Path(__file__).parent
    
    print("=" * 60)
    print("数据集格式验证工具")
    print("=" * 60)
    
    files_to_check = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    all_valid = True
    
    for filename in files_to_check:
        filepath = dataset_dir / filename
        if not validate_jsonl_file(filepath):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ 所有数据集文件格式正确！")
        print("\n下一步: 训练模型")
        print("  cd ../code")
        print("  ./train.sh")
    else:
        print("❌ 发现数据格式问题，请检查上述错误")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
