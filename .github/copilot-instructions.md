# CodeXGLUE AI Instructions

## Project Overview

CodeXGLUE is a benchmark suite for code intelligence tasks with 14 datasets across 10 tasks organized into 4 categories:
- `Code-Code/`: Clone detection, defect detection, cloze test, code completion, refinement, translation
- `Text-Code/`: NL code search, text-to-code generation
- `Code-Text/`: Code summarization
- `Text-Text/`: Documentation translation

Each task directory follows a consistent structure: `code/`, `dataset/`, `evaluator/`, `README.md`.

## Architecture & Key Patterns

### Task Structure Template
Each task implements the same workflow:
1. **Dataset preparation**: JSONL format in `dataset/` with train/valid/test splits
2. **Model training**: `code/run.py` with `train()`, `evaluate()`, `test()` functions
3. **Evaluation**: Standalone `evaluator/evaluator.py` script with `calculate_scores()`

### Model Architecture Convention
All classification/similarity tasks use the same pattern:
- Base encoder: RoBERTa from `transformers` (CodeBERT pretrained models)
- Custom head: Task-specific classifier (see `code/model.py`)
- Standard structure: `Model(encoder, config, tokenizer, args)` wrapper class

Example: `Code-Code/Clone-detection-BigCloneBench/code/model.py` shows the canonical pattern with `RobertaClassificationHead`.

### Data Format Standards
- **JSONL files**: One JSON object per line with task-specific fields
- **Common fields**: `idx` (index), `func`/`code` (source), `target`/`label` (ground truth)
- **Splits**: Consistent naming `train.txt`, `valid.txt`, `test.txt` or `.jsonl` variants
- **Literal normalization**: Frequent in datasets - uncommon strings/numbers replaced with `<STR_LIT>`, `<NUM_LIT>` tokens

## Critical Developer Workflows

### Running Fine-tuning
Each task uses the same command pattern from `code/` directory:
```bash
python run.py \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --do_train --do_eval --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=../saved_models \
    --block_size=512 \
    --train_batch_size=32 \
    --eval_batch_size=64 \
    --learning_rate=5e-5 \
    --num_train_epochs=10
```

### Running Evaluation
Evaluators are standalone and follow this pattern:
```bash
python evaluator/evaluator.py \
    -a evaluator/answers.txt \
    -p evaluator/predictions.txt
```

Different tasks use different metrics (see `evaluator/evaluator.py::calculate_scores()`):
- Clone detection: Precision, Recall, F1
- Defect detection: Accuracy
- Code translation: BLEU, Exact Match, CodeBLEU
- Code search: Mean Reciprocal Rank (MRR)

### CodeBLEU Metric
For generation tasks (translation, refinement, text-to-code), use CodeBLEU which combines:
- N-gram match (traditional BLEU)
- AST match (syntactic correctness)
- Data-flow match (logic correctness)

Located in `Code-Code/code-to-code-trans/evaluator/CodeBLEU/`.

## Project-Specific Conventions

### Pretrained Models
Three baseline models are used consistently:
- **CodeBERT** (`microsoft/codebert-base`): For understanding tasks (classification, search)
- **CodeGPT** (`microsoft/CodeGPT-small-java-adaptedGPT2`): For completion/generation
- **Encoder-Decoder**: For seq2seq generation

### Multi-GPU Training
Uses PyTorch DistributedDataParallel pattern:
```python
python -m torch.distributed.launch --nproc_per_node=$GPU_NUM run.py ...
```

### Model Classes Lookup
All tasks define `MODEL_CLASSES` dict mapping model type to (Config, Model, Tokenizer):
```python
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}
```

### Data Loading Pattern
Uses multiprocessing with caching for efficiency:
```python
pool = multiprocessing.Pool(cpu_cont)  # cpu_cont typically 16
examples = pool.map(get_example, data_tuples)
```

## Integration Points

### Dataset Sources
- BigCloneBench, POJ-104: Clone detection datasets
- Devign: Vulnerability detection dataset
- CodeSearchNet: Multi-language code search dataset
- py150, GitHub Java Corpus: Code completion datasets
- CONCODE: Text-to-code generation dataset

Download scripts are in each `dataset/` directory. Many use `gdown` for Google Drive files.

### Submission Workflow
To submit results to leaderboard:
1. Generate predictions on test set
2. Verify with dev set using local evaluator
3. Email `codexglue@microsoft.com` with predictions, model info, and paper details

Avoid multiple submissions in short timeframes to prevent p-hacking.

## Common Pitfalls

- **Token limits**: Most tasks use `block_size=512` for max sequence length
- **Literal normalization**: Test datasets have normalized literals - models must handle `<STR_LIT>`, `<NUM_LIT>` tokens
- **Function name masking**: AdvTest datasets replace function names/variables with special tokens to test generalization
- **Binary F1**: Clone detection uses binary F1, not macro F1 (updated 2021-09-13)
- **License**: Code is MIT, datasets use C-UDA license

## Domain-Specific Extensions: Particle Physics

### New Task: Code Review Intent Classification
Located in `Particle-Physics-Tasks/code-review-intent/`, this task evaluates models on particle physics code review understanding.

**Data Pipeline**:
1. Use `Git_crawler1` to crawl GitLab MR data from physics projects (ROOT, Geant4, CMSSW)
2. Run `dataset/preprocess_mr_data.py` to convert MR data to JSONL format
3. Train using standard CodeXGLUE workflow with `--num_labels=4`

**Quick Start**:
```bash
# Run complete pipeline
cd Particle-Physics-Tasks
./quickstart-physics-eval.sh your-physics-project

# Or manually:
cd Particle-Physics-Tasks/code-review-intent/dataset
python preprocess_mr_data.py --mr_data_dir=path/to/mr_data
```

**Physics-Specific Features**:
- Detects ROOT/Geant4 API usage patterns
- Classifies review intents: optimization (0), bug (1), clarification (2), approval (3)
- Handles C++/Python physics code with domain vocabulary

**Evaluation Plan**: See `Particle-Physics-Tasks/particle-physics-evaluation-plan.md` for comprehensive guide on evaluating code models for particle physics experiments.
