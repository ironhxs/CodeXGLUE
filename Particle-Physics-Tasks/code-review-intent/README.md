# CodeXGLUE -- 粒子物理代码审查意图分类

## 任务定义

给定代码变更和 MR 上下文，预测代码审查评论的意图类别。这是一个多分类任务，专门针对粒子物理实验软件开发场景。

### 意图类别
- **0 - 优化建议** (Optimization Suggestion): 建议改进代码性能、可读性或结构
- **1 - 错误报告** (Bug/Error Report): 指出代码中的错误、缺陷或潜在问题
- **2 - 澄清请求** (Clarification Request): 请求解释代码逻辑或设计决策
- **3 - 批准通过** (Approval/LGTM): 批准代码变更，准备合并

## 数据集构建

### 数据来源
使用 [Git_crawler1](https://github.com/15806487136/Git_crawler1) 爬取粒子物理相关项目的 GitLab Merge Requests：
- CERN ROOT 项目
- Geant4 模拟框架
- ALICE, ATLAS, CMS 实验代码
- 其他粒子物理软件项目

### 数据预处理

1. **爬取 MR 数据**:
```bash
cd Git_crawler1
# 示例：爬取 ROOT 项目
python crawler.py --project-id cern/root
```

2. **转换为 JSONL 格式**:
```bash
cd CodeXGLUE/Particle-Physics-Tasks/code-review-intent/dataset
python preprocess_mr_data.py \
    --mr_data_dir=../../../../Git_crawler1/mr_data \
    --output_dir=.
```

### 数据格式

每行一个 JSON 对象，包含以下字段：

```json
{
  "idx": 0,
  "code": "// File: src/TTree.cxx\nvoid TTree::Fill() {\n  fEntries++;\n  ...",
  "context": "Title: Fix memory leak in TTree::Fill\nDescription: Add proper cleanup...",
  "comment": "This might cause a segfault when fEntries exceeds MAX_INT",
  "target": 1,
  "mr_iid": 12345,
  "mr_url": "https://git.example.com/project/merge_requests/12345"
}
```

**字段说明**:
- `idx`: 样本索引
- `code`: 从 MR 中提取的代码变更（diff 中的新增代码）
- `context`: MR 标题和描述，提供上下文信息
- `comment`: 审查评论内容
- `target`: 意图类别 (0-3)
- `mr_iid`: MR 内部 ID
- `mr_url`: MR 链接（可选，用于追溯）

### 数据统计

数据集会自动按 80%/10%/10% 划分为训练集、验证集和测试集。

示例统计（需要实际爬取后生成）:
```
训练集: ~800 样本
验证集: ~100 样本  
测试集: ~100 样本
```

## 模型训练

### 依赖环境
```bash
pip install torch transformers
```

### Fine-tune CodeBERT

```bash
cd ../code
python run.py \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --do_train --do_eval --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=../saved_models \
    --num_labels=4 \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5
```

### 参数说明
- `--num_labels=4`: 4 个意图类别
- `--block_size=512`: 最大序列长度
- `--train_batch_size=16`: 训练批次大小（根据 GPU 显存调整）
- `--learning_rate=2e-5`: 学习率
- `--num_train_epochs=5`: 训练轮数

## 评估

### 运行评估器
```bash
cd ../evaluator
python evaluator.py \
    -a ../dataset/test.jsonl \
    -p ../saved_models/predictions.txt
```

### 评估指标
- **Accuracy**: 整体准确率
- **Macro F1**: 各类别 F1 分数的平均值
- **Per-class Precision/Recall/F1**: 每个类别的详细指标

### 示例输出
```json
{
  "Accuracy": 0.75,
  "Macro F1": 0.72,
  "Class 0 (Optimization)": {"Precision": 0.70, "Recall": 0.68, "F1": 0.69},
  "Class 1 (Bug/Error)": {"Precision": 0.80, "Recall": 0.82, "F1": 0.81},
  "Class 2 (Clarification)": {"Precision": 0.65, "Recall": 0.60, "F1": 0.62},
  "Class 3 (Approval)": {"Precision": 0.85, "Recall": 0.88, "F1": 0.86}
}
```

## 粒子物理特定特征

本任务捕获了粒子物理软件开发的特点：

1. **领域特定 API**:
   - ROOT 框架: `TTree`, `TBranch`, `TH1F`
   - Geant4: `G4Step`, `G4Track`, `G4Event`
   - 分析框架: `CMSSW`, `Athena`, `GaudiKernel`

2. **物理概念**:
   - 物理量: energy, momentum, mass
   - 单位: GeV, TeV, cm, ns
   - 算法: Monte Carlo, reconstruction, trigger

3. **性能关注**:
   - 内存管理（常见于 C++ 代码）
   - 并行计算优化
   - GPU 加速

## 应用场景

训练好的模型可用于：

1. **代码审查助手**: 自动分类审查评论，帮助开发者优先处理错误报告
2. **智能通知**: 根据意图类别发送不同优先级的通知
3. **代码质量分析**: 统计项目中各类审查意图的分布，识别代码质量问题
4. **开发者推荐**: 根据评论意图推荐合适的审查者

## 扩展方向

1. **多任务学习**: 同时预测意图类别和严重程度
2. **生成式任务**: 给定代码变更，生成审查评论
3. **跨项目迁移**: 在一个项目上训练，在另一个项目上测试泛化能力
4. **时序分析**: 分析审查意图随时间的变化趋势

## 引用

如果使用本数据集，请引用：

```bibtex
@dataset{particle_physics_code_review,
  title={Particle Physics Code Review Intent Classification Dataset},
  author={Your Name},
  year={2025},
  note={Based on GitLab MR data from particle physics projects}
}
```

## 许可证

- 代码: MIT License
- 数据集: 遵循源项目的开源许可证

## 联系方式

如有问题或建议，请通过 Issue 或邮件联系。
