#!/bin/bash
# ============================================================
# 粒子物理代码模型评估 - 一键运行脚本
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "粒子物理代码模型评估"
echo "============================================================"

# 检查参数
MR_DATA_DIR="${1:-./Git_crawler1/mr_data}"
TASK="${2:-all}"

if [ ! -d "$MR_DATA_DIR" ]; then
    echo -e "${RED}Error: MR data directory not found: $MR_DATA_DIR${NC}"
    echo ""
    echo "Usage: $0 <mr_data_dir> [task]"
    echo "  mr_data_dir: Path to MR data directory (containing mr_*.json files)"
    echo "  task: all, summarization, completion, defect"
    exit 1
fi

echo ""
echo -e "${GREEN}MR Data Directory: $MR_DATA_DIR${NC}"
echo -e "${GREEN}Task: $TASK${NC}"
echo ""

# Step 1: 数据预处理
echo "============================================================"
echo "Step 1: 数据预处理"
echo "============================================================"

python preprocess_all_tasks.py \
    --mr_data_dir="$MR_DATA_DIR" \
    --output_base_dir="." \
    --tasks="$TASK"

echo ""
echo -e "${GREEN}✓ 数据预处理完成${NC}"
echo ""

# 检查是否有足够数据
check_data() {
    local task_dir="$1"
    local train_file="$task_dir/dataset/train.jsonl"
    
    if [ ! -f "$train_file" ]; then
        echo -e "${YELLOW}Warning: No training data for $task_dir${NC}"
        return 1
    fi
    
    local count=$(wc -l < "$train_file")
    if [ "$count" -lt 10 ]; then
        echo -e "${YELLOW}Warning: Only $count samples for $task_dir, skipping training${NC}"
        return 1
    fi
    
    echo -e "${GREEN}$task_dir: $count training samples${NC}"
    return 0
}

# Step 2: 训练模型 (可选)
echo "============================================================"
echo "Step 2: 模型训练 (需要 GPU)"
echo "============================================================"

# 检查是否有 GPU
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}GPU detected, training enabled${NC}"
    GPU_AVAILABLE=1
else
    echo -e "${YELLOW}No GPU detected, skipping training${NC}"
    GPU_AVAILABLE=0
fi

# 任务1: 代码摘要生成
if [[ "$TASK" == "all" || "$TASK" == "summarization" ]]; then
    if check_data "code-summarization"; then
        if [ "$GPU_AVAILABLE" -eq 1 ]; then
            echo ""
            echo "Training Code Summarization model..."
            cd code-summarization/code
            python run.py \
                --do_train --do_eval --do_test \
                --train_data_file=../dataset/train.jsonl \
                --eval_data_file=../dataset/valid.jsonl \
                --test_data_file=../dataset/test.jsonl \
                --output_dir=../saved_models \
                --model_name_or_path=microsoft/codebert-base \
                --max_source_length=256 \
                --max_target_length=64 \
                --beam_size=5 \
                --train_batch_size=8 \
                --eval_batch_size=16 \
                --learning_rate=5e-5 \
                --num_train_epochs=5
            cd ../..
        fi
    fi
fi

# 任务3: 代码补全
if [[ "$TASK" == "all" || "$TASK" == "completion" ]]; then
    if check_data "code-completion"; then
        if [ "$GPU_AVAILABLE" -eq 1 ]; then
            echo ""
            echo "Training Code Completion model..."
            cd code-completion/code
            python run.py \
                --do_train --do_eval --do_test \
                --train_data_file=../dataset/train.jsonl \
                --eval_data_file=../dataset/valid.jsonl \
                --test_data_file=../dataset/test.jsonl \
                --output_dir=../saved_models \
                --model_name_or_path=gpt2 \
                --block_size=256 \
                --max_gen_length=30 \
                --beam_size=5 \
                --train_batch_size=8 \
                --learning_rate=5e-5 \
                --num_train_epochs=5
            cd ../..
        fi
    fi
fi

# 任务4: 缺陷检测
if [[ "$TASK" == "all" || "$TASK" == "defect" ]]; then
    if check_data "defect-detection"; then
        if [ "$GPU_AVAILABLE" -eq 1 ]; then
            echo ""
            echo "Training Defect Detection model..."
            cd defect-detection/code
            python run.py \
                --do_train --do_eval --do_test \
                --train_data_file=../dataset/train.jsonl \
                --eval_data_file=../dataset/valid.jsonl \
                --test_data_file=../dataset/test.jsonl \
                --output_dir=../saved_models \
                --model_name_or_path=microsoft/codebert-base \
                --block_size=512 \
                --train_batch_size=16 \
                --learning_rate=2e-5 \
                --num_train_epochs=5
            cd ../..
        fi
    fi
fi

echo ""
echo -e "${GREEN}✓ 训练完成${NC}"
echo ""

# Step 3: 评估结果
echo "============================================================"
echo "Step 3: 评估结果"
echo "============================================================"

if [[ "$TASK" == "all" || "$TASK" == "summarization" ]]; then
    if [ -f "code-summarization/saved_models/predictions.txt" ]; then
        echo ""
        echo "=== Code Summarization Results ==="
        python code-summarization/evaluator/evaluator.py \
            -a code-summarization/saved_models/gold.txt \
            -p code-summarization/saved_models/predictions.txt
    fi
fi

if [[ "$TASK" == "all" || "$TASK" == "completion" ]]; then
    if [ -f "code-completion/saved_models/predictions.txt" ]; then
        echo ""
        echo "=== Code Completion Results ==="
        python code-completion/evaluator/evaluator.py \
            -a code-completion/saved_models/gold.txt \
            -p code-completion/saved_models/predictions.txt
    fi
fi

if [[ "$TASK" == "all" || "$TASK" == "defect" ]]; then
    if [ -f "defect-detection/saved_models/predictions.txt" ]; then
        echo ""
        echo "=== Defect Detection Results ==="
        python defect-detection/evaluator/evaluator.py \
            -a defect-detection/saved_models/gold.txt \
            -p defect-detection/saved_models/predictions.txt \
            --probs defect-detection/saved_models/probabilities.txt
    fi
fi

echo ""
echo "============================================================"
echo -e "${GREEN}评估完成!${NC}"
echo "============================================================"
