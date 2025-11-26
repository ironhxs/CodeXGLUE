#!/bin/bash
# 粒子物理代码审查意图分类 - 示例训练脚本

# 设置变量
DATA_DIR="../dataset"
OUTPUT_DIR="../saved_models"
MODEL_NAME="microsoft/codebert-base"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练模型
python run.py \
    --model_type=roberta \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$DATA_DIR/train.jsonl \
    --eval_data_file=$DATA_DIR/valid.jsonl \
    --test_data_file=$DATA_DIR/test.jsonl \
    --output_dir=$OUTPUT_DIR \
    --num_labels=4 \
    --block_size=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --seed=42

echo "训练完成！"
echo "预测结果保存在: $OUTPUT_DIR/predictions.txt"
echo ""
echo "运行评估器:"
echo "cd ../evaluator"
echo "python evaluator.py -a ../dataset/test.jsonl -p ../saved_models/predictions.txt"
