#!/usr/bin/env bash
# Source me to enter the SATA training env:
#   source scripts/sata-env.sh
# (Don't execute — `conda activate` only works in the calling shell.)
#
# Sets up: sata conda env (Python 3.8 + Isaac Gym + SATA stack), then cd into
# the SATA legged_gym dir ready for train.py / play.py.

if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "sata-env.sh must be sourced, not executed." >&2
  exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sata
cd "$HOME/workspace/SATA/legged_gym/legged_gym"

cat <<EOF
[sata] env active. Suggested commands:
  # full reference run (4096 envs, 3000 iters)
  WANDB_MODE=disabled python scripts/train.py --task=go2_torque --headless --num_envs=4096 --max_iterations=3000

  # tensorboard
  tensorboard --logdir ~/workspace/SATA/legged_gym/logs --port 6006 --bind_all
EOF
