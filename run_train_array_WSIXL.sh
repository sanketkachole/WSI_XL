#!/bin/bash
#SBATCH --job-name=wsi_xl_train
#SBATCH --account=r01753
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

export PS1=${PS1:-}
export TOKENIZERS_PARALLELISM=false

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate WSI_XL_ENV   # <-- change to your actual env name

# go to repo root (where main.py is)
cd /N/project/Sanket_Slate_Project/3_modelling/MICCAI_Slide_XL

mkdir -p logs

FOLD_ID="${SLURM_ARRAY_TASK_ID}"
FOLD_DIR=$(printf "fold_%02d" "${FOLD_ID}")

# --- REQUIRED PATHS (edit these) ---
DATA_ROOT="/N/project/Sanket_Slate_Project/3_modelling/MICCAI_Slide_XL/dataset_master/meta"
TEXT_TOK="/N/project/Sanket_Slate_Project/2_feature_extractors/text_feature_extractors/distilbert_text_token_level_embeddings/text_report_distilbert_token_embeds.npy"
TEXT_MSK="/N/project/Sanket_Slate_Project/2_feature_extractors/text_feature_extractors/distilbert_text_token_level_embeddings/text_report_distilbert_attention_mask.npy"
OUT_DIR="/N/project/Sanket_Slate_Project/3_modelling/MICCAI_Slide_XL/runs/${FOLD_DIR}"

# --- TRAINING ---
python main.py \
  --data_root "${DATA_ROOT}" \
  --splits_subdir "splits/${FOLD_DIR}" \
  --train_csv "train.csv" \
  --val_csv "val.csv" \
  --test_csv "test.csv" \
  --label_cols "survival_event" \
  --task_type binary \
  --d_v 2560 \
  --d_t 768 \
  --d_model 512 \
  --n_layers 4 \
  --n_heads 8 \
  --d_ff 2048 \
  --chunk_size 4096 \
  --max_report_len 512 \
  --text_token_embeds_npy "${TEXT_TOK}" \
  --text_attention_mask_npy "${TEXT_MSK}" \
  --epochs 5 \
  --lr 1e-4 \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda \
  --amp \
  --out_dir "${OUT_DIR}" \
  --save_best
