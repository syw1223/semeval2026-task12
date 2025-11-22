# SemEval 2026 Task 12 — Multi-Label MCQ (BERT TopK + Calibration)

Reproducible codebase for our SemEval-2026 Task 12 system.
Includes training, official evaluation (exact/partial/official), and post-hoc threshold calibration.

## Environment
conda env create -f env/environment.yml
conda activate semeval12
# or: pip install -r env/requirements.txt

## Data Layout (not included)
data/
  train/
    questions.jsonl
    docs.json
  dev/
    questions.jsonl
    docs.json

## Train + Dev Predict
python scripts/SemEval_Simple_TopK3.py \
  --train_jsonl data/train/questions.jsonl \
  --train_docs_json data/train/docs.json \
  --dev_jsonl   data/dev/questions.jsonl \
  --dev_docs_json   data/dev/docs.json \
  --save_dir outputs_bert_topk3 \
  --backbone bert-base-uncased \
  --epochs 3 --batch_size 1 \
  --max_len 384 --window 224 --stride 144 --max_windows 6 \
  --agg topk --topk_k 3 \
  --pos_weight --amp

## Post-hoc Threshold Sweep (official metric)
python scripts/posthoc_delta_sweep.py \
  --pred_in outputs_bert_topk3/dev_predictions.json \
  --gold    data/dev/questions.jsonl \
  --calib   outputs_bert_topk3/val_calibration.json \
  --delta_min -0.05 --delta_max 0.05 --delta_step 0.01 \
  --select official

## Official Evaluation
python scripts/evaluate_dev_official.py \
  --pred outputs_bert_topk3/dev_predictions_rethr.json \
  --gold data/dev/questions.jsonl

## Repo Layout
- scripts/  training/eval/calibration scripts
- configs/  typical hyper-params
- experiments/  command + metrics + notes per run
- outputs/  only lightweight metrics/plots
- docs/  methodology and experiment logs
- env/  reproducible environment files

## License
MIT (see LICENSE).
