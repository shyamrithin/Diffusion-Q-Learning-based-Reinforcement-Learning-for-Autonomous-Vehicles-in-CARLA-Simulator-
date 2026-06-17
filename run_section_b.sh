#!/usr/bin/env bash
# ==========================================================
# run_section_b.sh
# Section B (traffic-density) evaluation batch for DQL-E paper.
#
# Runs all three agents across 4 traffic densities, all routes,
# n=5 episodes per cell. DQL-E first (per request), then SAC,
# then PPO. Each cell logged; a failing cell is recorded and
# skipped so the batch keeps going overnight unattended.
#
# Matrix: 3 agents x 4 densities x (route 'all' = R1,R2,R3) x 5 ep
#         = 180 episodes total.
#
# Honest scoring (applied later in agg_b.py, NOT here):
#   R3 straight -> completion + trajectory error (fair, 1 path)
#   R1/R2       -> safety only (collision/offroad/stuck;
#                  exit-independent, agents diverge legally)
#
# Usage:
#   conda activate diffusioncarla
#   # make sure CARLA server is running in another terminal:
#   #   ./CarlaUE4.sh -quality-level=Low -RenderOffScreen
#   bash run_section_b.sh
#
# Resumable: cells whose summary CSV already exists are skipped,
# so if it crashes you can re-launch and it continues.
# ==========================================================

set -u
cd ~/Carla/RLCarla

AGENTS=("dqle" "sac" "ppo")          # DQL-E first
DENSITIES=("empty" "light" "medium" "heavy")
EPISODES=5
CKPT_ID=9999
MAX_STEPS=1000

LOG_DIR="results/section_b_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"

echo "=== Section B eval batch started $(date) ===" | tee "$MASTER_LOG"
echo "Agents: ${AGENTS[*]} | Densities: ${DENSITIES[*]} | n=$EPISODES" \
    | tee -a "$MASTER_LOG"

TOTAL=$(( ${#AGENTS[@]} * ${#DENSITIES[@]} ))
i=0
FAILED=()

for agent in "${AGENTS[@]}"; do
  for traffic in "${DENSITIES[@]}"; do
    i=$((i+1))
    cell="${agent}_${traffic}"
    cell_log="$LOG_DIR/${cell}.log"
    echo "" | tee -a "$MASTER_LOG"
    echo "[$i/$TOTAL] $(date +%H:%M:%S)  RUN  agent=$agent traffic=$traffic" \
        | tee -a "$MASTER_LOG"

    # Resume guard: skip if this cell's summary already exists
    if ls results/${cell}*/ALL_summary.csv >/dev/null 2>&1; then
      echo "    -> already done (summary exists), skipping" \
          | tee -a "$MASTER_LOG"
      continue
    fi

    python3 record_eval.py \
        --agent "$agent" \
        --traffic "$traffic" \
        --route all \
        --episodes "$EPISODES" \
        --ckpt_id "$CKPT_ID" \
        --max_steps "$MAX_STEPS" \
        > "$cell_log" 2>&1

    rc=$?
    if [ $rc -ne 0 ]; then
      echo "    -> FAILED (exit $rc) — see $cell_log" | tee -a "$MASTER_LOG"
      FAILED+=("$cell")
    else
      echo "    -> OK" | tee -a "$MASTER_LOG"
    fi
  done
done

echo "" | tee -a "$MASTER_LOG"
echo "=== Batch finished $(date) ===" | tee -a "$MASTER_LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED cells: ${FAILED[*]}" | tee -a "$MASTER_LOG"
  echo "Re-launch the script to retry only the failed/missing cells." \
      | tee -a "$MASTER_LOG"
else
  echo "All ${TOTAL} cells completed. Next: python3 agg_b.py" \
      | tee -a "$MASTER_LOG"
fi