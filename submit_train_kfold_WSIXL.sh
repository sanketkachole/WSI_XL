#!/bin/bash
set -euo pipefail

# number of folds you created: fold_00 ... fold_04 => 5 folds
K=5

mkdir -p logs
sbatch --array=0-$((K-1)) run_train_array_WSIXL.sh
