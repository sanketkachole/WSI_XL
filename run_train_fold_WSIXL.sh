#!/bin/bash
#SBATCH --job-name=wsixl
#SBATCH --account=r01753
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/wsixl_%A_%a.out
#SBATCH --error=logs/wsixl_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate WSIXL

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_Slide_XL

mkdir -p logs

FOLD_ID="${SLURM_ARRAY_TASK_ID}"
FOLD_DIR=$(printf "fold_%02d" "${FOLD_ID}")

DATA_ROOT="/N/project/Sanket_Slate_Project/4_modelling/MICCAI_Slide_XL/dataset_master/meta"
TEXT_TOK="/N/project/Sanket_Slate_Project/2_feature_extractors/text_feature_extractors/distilbert_text_token_level_embeddings/text_report_distilbert_token_embeds.npy"
TEXT_MSK="/N/project/Sanket_Slate_Project/2_feature_extractors/text_feature_extractors/distilbert_text_token_level_embeddings/text_report_distilbert_attention_mask.npy"
OUT_DIR="/N/project/Sanket_Slate_Project/4_modelling/MICCAI_Slide_XL/runs/${FOLD_DIR}"

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
  --chunk_size 2048 \
  --max_report_len 256 \
  --text_token_embeds_npy "${TEXT_TOK}" \
  --text_attention_mask_npy "${TEXT_MSK}" \
  --epochs 5 \
  --lr 1e-4 \
  --batch_size 1 \
  --num_workers 2 \
  --device cuda \
  --amp \
  --out_dir "${OUT_DIR}" \
  --save_best
