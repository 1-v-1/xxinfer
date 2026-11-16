#!/usr/bin/env bash
# mtp_decode_breakdown_probe —— 一个 MTP 周期的分段耗时 + 开/关端到端 A/B
#
# @case d2#1 基线 A/B 表（MTP 关 vs 开，decode tok/s）
# @case d2#2 一个周期的分段耗时表（step1 / draft / verify / 其他，各占比）
#
# 同条件交替采样（ROUNDS 轮，默认 5；D12 要求交替，不允许各跑一半再比）。
# 产物落 $XINFER_PROBE_DIR/mtp-decode-breakdown/（产物不入 git，结论入 git）。
#
# 用法：ROUNDS=5 TOKENS=512 bash tests/probes/mtp_decode_breakdown_probe.sh
# 覆盖：BIN MODEL K ROUNDS TOKENS MML OFFLOAD PROMPT XINFER_PROBE_DIR
set -uo pipefail

BIN=${BIN:-/workspace/xxinfer/target/release/xinfer}
MODEL=${MODEL:-/data/qwen3_5_8layer}
K=${K:-3}
ROUNDS=${ROUNDS:-5}
TOKENS=${TOKENS:-512}
MML=${MML:-8192}
# 并行度：TP2 目前踩在 oneshot allreduce 的非法地址上（见 §1），本件先用 TP1 取数。
DEVS=${DEVS:-0}
DTYPE=${DTYPE:-bf16}
# 只在模型需要时传（171GB 的 MoE 检查点才需要 auto）。
OFFLOAD=${OFFLOAD:-}
OUT="${XINFER_PROBE_DIR:-/workspace/probe-out}/mtp-decode-breakdown"
# 固定 prompt：同一句话，贪心，这样两轮之间只有 MTP 开关一个变量。
PROMPT=${PROMPT:-'Explain in one short paragraph why a cache-bounded GPU spends most of its decode time moving weights instead of computing them.'}

mkdir -p "$OUT"
{
  echo "model=$MODEL  dtype=$DTYPE  devices=$DEVS  max_model_len=$MML  moe_offload=${OFFLOAD:-unset}"
  echo "max_tokens=$TOKENS  temperature=0  ignore_eos=1  k=$K  rounds=$ROUNDS  extra_args=${BASE:-none}"
  echo "clocks=$(nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm --format=csv,noheader 2>/dev/null | tr '\n' ' ') (未锁：容器内无权限改时钟)"
} > "$OUT/conditions.txt"

# 一次完整运行。第 1 参 = 标签，其余 = 额外 CLI 参数。
run() {
  local label=$1; shift
  local envp=(XINFER_LOG_LEVEL=debug XINFER_DIAG_MTP=1)
  [[ -n "$OFFLOAD" ]] && envp+=("XINFER_MOE_OFFLOAD=$OFFLOAD")
  env "${envp[@]}" "$BIN" --m "$MODEL" --dtype "$DTYPE" --d "$DEVS" --max-model-len "$MML" ${BASE:-} \
      --prompts "$PROMPT" --max-tokens "$TOKENS" --temperature 0 --ignore-eos \
      "$@" > "$OUT/$label.log" 2>&1
  local rc=$? line n t rate
  line=$(sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$label.log" | grep "Decoded tokens:" | tail -1)
  n=$(sed -n 's/.*Decoded tokens: \([0-9]*\) in.*/\1/p' <<<"$line")
  t=$(sed -n 's/.* in \([0-9.]*\)s .*/\1/p' <<<"$line")
  rate=$(sed -n 's/.*(\([0-9.]*\) tokens\/s).*/\1/p' <<<"$line")
  local graphs
  graphs=$(grep -c "graph captured\|Capturing" "$OUT/$label.log")
  echo "$label rc=$rc tokens=$n time=${t}s tok_per_s=$rate graphs=$graphs" >> "$OUT/results.tsv"
  # 分段：只在 MTP 开时存在。
  if grep -q "MTP step1" "$OUT/$label.log"; then
    sed -e 's/\x1b\[[0-9;]*m//g' "$OUT/$label.log" | awk -v tag="$label" '
        /MTP step1/  {s+=$(NF-1)+0; ns++}
        /MTP draft/  {d+=$(NF-1)+0; nd++}
        /MTP verify/{v+=$(NF-1)+0; nv++}
        END{printf "%s cycles=%d step1=%.2fms draft=%.2fms verify=%.2fms\n",
            tag, nv, s/ns, d/nd, v/nv}' >> "$OUT/segments.tsv"
  fi
}

: > "$OUT/results.tsv"; : > "$OUT/segments.tsv"
for ((i=1; i<=ROUNDS; i++)); do
  echo "==> round $i: off"; run "off_$i"
  echo "==> round $i: on";  run "on_$i" --num-speculative-tokens "$K"
done

echo "== A/B (d2#1) =="; column -t "$OUT/results.tsv" 2>/dev/null || cat "$OUT/results.tsv"
echo "== segments (d2#2) =="; column -t "$OUT/segments.tsv" 2>/dev/null || cat "$OUT/segments.tsv"
