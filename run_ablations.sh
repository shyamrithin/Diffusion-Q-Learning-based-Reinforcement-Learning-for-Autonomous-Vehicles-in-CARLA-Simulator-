#!/usr/bin/env bash
# ==========================================================
# run_ablations.sh
# Overnight ablation runner for the DQL-E Section D study.
#
# Runs the two ablation trainings SEQUENTIALLY (never in
# parallel — they share one GPU and one CARLA server):
#   1) no_eta      : eta held at ETA_END, no warmup ramp
#   2) no_pretrain : PRE_TRAIN_CRITIC_STEPS = 0
# Each writes to checkpoints_ablate_<tag> / runs/ablate_<tag>
# and a timestamped log under ablation_logs/. v14/v15b are
# never touched. The reference curves come from existing
# v15b/v14 runs (no run needed).
#
# USAGE:
#   1. Start the CARLA server FIRST (separate terminal):
#        ./CarlaUE4.sh -RenderOffScreen   (or your usual command)
#   2. Then, from ~/Carla/RLCarla with the conda env active:
#        nohup bash run_ablations.sh > ablation_logs/run.out 2>&1 &
#      (nohup + & keeps it alive after you log out / close the terminal)
#   3. Check progress in the morning:
#        tail -f ablation_logs/run.out
# ==========================================================
set -u
cd "$(dirname "$0")"            # run from the script's folder (RLCarla)

EPISODES=700
mkdir -p ablation_logs
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=================================================="
echo "[ABLATION RUNNER] start $(date)"
echo "  episodes per run : $EPISODES"
echo "  python           : $(which python3)"
echo "  conda env        : ${CONDA_DEFAULT_ENV:-<none>}"
echo "=================================================="

run_one () {
    local mode="$1"; local tag="$1"
    local log="ablation_logs/${tag}_${STAMP}.log"
    echo ""
    echo ">>> [$(date)] starting ablation: $mode  ->  $log"
    python3 train_ablation.py \
        --ablation "$mode" \
        --episodes "$EPISODES" \
        --tag "$tag" \
        > "$log" 2>&1
    local rc=$?
    echo ">>> [$(date)] ablation '$mode' finished, exit code $rc"
    return $rc
}

# Run sequentially. If the first crashes we STILL try the second,
# so you don't lose the whole night to one failure.
run_one "no_eta"      || echo "!! no_eta run exited non-zero (continuing)"
sleep 10
run_one "no_pretrain" || echo "!! no_pretrain run exited non-zero (continuing)"

echo ""
echo "=================================================="
echo "[ABLATION RUNNER] all done $(date)"
echo "  checkpoints: checkpoints_ablate_no_eta , checkpoints_ablate_no_pretrain"
echo "  tb logs    : runs/ablate_no_eta , runs/ablate_no_pretrain"
echo "  per-run logs: ablation_logs/"
echo "=================================================="
