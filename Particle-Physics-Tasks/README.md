# 粒子物理代码模型评估框架

基于 CodeXGLUE 框架，评估代码模型在粒子物理实验领域的能力。

> **使用场景**: 其他成员训练好模型后，用此框架在粒子物理领域数据上进行评估。

## 🎯 工作流程

```
1. 爬取 MR 数据 (Git_crawler1)
2. 预处理成测试集 (preprocess_all_tasks.py)
3. 加载训练好的模型
4. 在测试集上评估
5. 输出评估报告
```

## 📁 项目结构

```
Particle-Physics-Tasks/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── preprocess_all_tasks.py      # 数据预处理脚本
├── run_all.sh                   # 一键运行脚本
│
├── Git_crawler1/                # GitLab MR 爬虫
│   └── crawler.py
│
├── code-summarization/          # 任务1: 代码摘要生成
│   ├── code/
│   │   ├── model.py             # Seq2Seq 模型
│   │   └── run.py               # 训练脚本
│   ├── dataset/                 # 数据集
│   └── evaluator/
│       └── evaluator.py         # 评估器
│
├── code-completion/             # 任务3: 代码补全
│   ├── code/
│   │   ├── model.py             # GPT 模型
│   │   └── run.py
│   ├── dataset/
│   └── evaluator/
│       └── evaluator.py
│
└── defect-detection/            # 任务4: 缺陷检测
    ├── code/
    │   ├── model.py             # 分类模型
    │   └── run.py
    ├── dataset/
    └── evaluator/
        └── evaluator.py
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

使用爬虫获取 GitLab MR 数据：
```bash
cd Git_crawler1
python crawler.py --project-url https://gitlab.com/your-project
```

### 3. 评估模式（推荐）

**你的场景：其他成员训练好模型 → 你用爬来的数据评估**

```bash
# Step 1: 只生成测试集
python preprocess_all_tasks.py \
    --mr_data_dir=./Git_crawler1/mr_data \
    --eval_only

# Step 2: 加载外部模型进行评估
cd code-summarization/code
python run.py \
    --do_test \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=../saved_models \
    --model_name_or_path=/path/to/trained/model  # 其他成员训练好的模型

# Step 3: 计算指标
python ../evaluator/evaluator.py \
    -a ../saved_models/gold.txt \
    -p ../saved_models/predictions.txt
```

### 4. 完整模式（训练+评估）

如果需要自己训练：
```bash
# 生成完整数据集 (train/valid/test)
python preprocess_all_tasks.py --mr_data_dir=./Git_crawler1/mr_data

# 训练并评估
cd code-summarization/code
python run.py \
    --do_train --do_eval --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=../saved_models
```

---

## 📊 三个评估任务

### 任务1: 代码摘要生成 (Code Summarization)

**目标**: 给定代码变更 (diff)，生成描述性摘要

| 项目 | 说明 |
|------|------|
| 输入 | 代码 diff |
| 输出 | MR title (摘要) |
| 模型 | CodeBERT + Transformer Decoder |
| 指标 | BLEU, ROUGE-L |

**数据构造**:
```
输入: - void process() { old_code }
      + void process() { new_code }  
输出: "Fix memory leak in process function"
```

### 任务3: 代码补全 (Code Completion)

**目标**: 给定代码上下文，预测后续代码

| 项目 | 说明 |
|------|------|
| 输入 | 代码前缀 |
| 输出 | 代码后缀 |
| 模型 | GPT-2 / CodeGPT |
| 指标 | Edit Similarity, Exact Match |

**数据构造**:
```
输入: if (buffer == nullptr) {
输出: return -1; }
```

### 任务4: 缺陷检测 (Defect Detection)

**目标**: 判断代码是否包含缺陷

| 项目 | 说明 |
|------|------|
| 输入 | 代码片段 |
| 输出 | 0 (无缺陷) / 1 (有缺陷) |
| 模型 | CodeBERT + Classifier |
| 指标 | Accuracy, F1, AUC-ROC |

**数据构造**:
- 正样本 (label=1): bug-fix MR 中被删除的代码
- 负样本 (label=0): 非 bug-fix MR 中新增的代码

---

## 🔄 更换模型

只需修改 `--model_name_or_path` 参数：

```bash
# CodeBERT (默认)
--model_name_or_path=microsoft/codebert-base

# GraphCodeBERT
--model_name_or_path=microsoft/graphcodebert-base

# UniXcoder
--model_name_or_path=microsoft/unixcoder-base

# CodeGPT (代码补全)
--model_name_or_path=microsoft/CodeGPT-small-py

# StarCoder (代码补全)
--model_name_or_path=bigcode/starcoderbase-1b
```

---

## 📈 评估流程

```
┌─────────────────┐
│  MR 原始数据     │  ← Git_crawler1 爬取
│  (JSON files)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  preprocess     │  ← 构造 (input, ground_truth) 对
│  _all_tasks.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  train/valid/   │
│  test.jsonl     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  run.py         │  ← 模型训练 & 推理
│  (训练/测试)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  evaluator.py   │  ← 计算评估指标
│  (评估)         │
└─────────────────┘
```

---

## 📋 数据格式

### 代码摘要 (train.jsonl)
```json
{"idx": 0, "code": "- old\n+ new", "summary": "Fix bug", "mr_iid": "123"}
```

### 代码补全 (train.jsonl)
```json
{"idx": 0, "context": "if (x ==", "target": " nullptr) return;", "file_path": "src/main.cpp"}
```

### 缺陷检测 (train.jsonl)
```json
{"idx": 0, "func": "void foo() { ... }", "target": 1, "mr_iid": "456"}
```

---

## ⚙️ 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 预训练模型 | `microsoft/codebert-base` |
| `--train_batch_size` | 训练批次 | 8 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--num_train_epochs` | 训练轮数 | 5 |
| `--block_size` | 最大序列长度 | 512 |
| `--do_train` | 启用训练 | - |
| `--do_eval` | 启用验证 | - |
| `--do_test` | 启用测试 | - |

---

## 🔗 相关链接

- [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) - 微软代码智能基准
- [CodeBERT](https://huggingface.co/microsoft/codebert-base) - 预训练代码模型
- [Transformers](https://huggingface.co/docs/transformers) - HuggingFace 模型库
