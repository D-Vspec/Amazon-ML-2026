#!/bin/bash
# Full run (docs/full_run_report.md): fine-tune on 8 sets, score on sets 2/3, then the test submission.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
cp .env.example .env  # replaces any local .env: the exports below are the whole configuration
export PYTHONUNBUFFERED=1
export DATA_DIR=6ab10eb3b23ba_student_resource/student_resource/dataset
export N_SETS=10 SPLIT=train NROWS= BLOCKER=tfidf+embedding BLOCK_K=20 DEVICE=cuda CACHE_DIR=data/cache
export MATCHER=cross_encoder TRAIN_SETS= TUNE_SETS=2 MATCHER_MODEL=
export CROSS_ENCODER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 CE_MODEL_DIR=models/ce_full
export CE_TRAIN_SETS=0,1,4,5,6,7,8,9 CE_TRAIN_PAIRS=12000000

echo "=== stage 1: fine-tune on sets $CE_TRAIN_SETS, threshold on set 2, F0.5 on set 3 ($(date)) ==="
SETS=3 uv run python main.py 2>&1 | tee logs/stage1.log

echo "=== stage 2: test submission ($(date)) ==="
SETS= SPLIT=test uv run python main.py 2>&1 | tee logs/stage2.log

echo "=== validate ($(date)) ==="
uv run python utils/validate_submission.py -t "$DATA_DIR/test" --check-ids 2>&1 | tee logs/validate.log
echo "=== done ($(date)) ==="
