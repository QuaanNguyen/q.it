#!/bin/sh
# Drive the q.it speech path end to end and capture evidence.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
evidence_root="$root/.qit-evidence/verify"

cleanup_dir() {
  if [ -f "$1/pids" ]; then
    while read -r pid; do
      kill "$pid" 2>/dev/null || true
    done < "$1/pids"
    sleep 0.3
    while read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done < "$1/pids"
  fi
}

if [ "${1:-}" = "--cleanup" ]; then
  newest=$(ls -1dt "$evidence_root"/*/ 2>/dev/null | head -1 || true)
  if [ -z "$newest" ]; then
    echo "no evidence directory to clean"
    exit 0
  fi
  cleanup_dir "${newest%/}"
  echo "stopped processes recorded in ${newest%/}"
  exit 0
fi

cargo build --locked -p qit
binary="$root/target/debug/qit"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$evidence_root/$stamp"
scratch="$out/scratch"
state="$scratch/state"
mkdir -p "$out" "$scratch"
trap 'cleanup_dir "$out"' EXIT INT TERM

model="parakeet-fixture"
python3 "$here/fixture_provider.py" --port-file "$scratch/port" 2>"$out/provider.log" &
echo $! >>"$out/pids"
for _ in $(seq 1 100); do
  [ -s "$scratch/port" ] && break
  sleep 0.1
done
port=$(cat "$scratch/port")
base="http://127.0.0.1:$port"

set +e
QIT_HOME="$state" "$binary" run core/speech-to-text-smoke \
  --provider local-cpu --model "$model" \
  --base-url "$base/$model" --iterations 2 --warmups 1 \
  >"$out/run.out" 2>"$out/run.err"
echo $? >"$out/run.status"
set -e

QIT_HOME="$state" "$binary" results --limit 5 >"$out/results.out" 2>&1
QIT_HOME="$state" "$binary" export --format jsonl --output "$out/export.jsonl"

QIT_HOME="$state" "$binary" dashboard --port 0 >"$out/dashboard.log" 2>&1 &
dashboard_pid=$!
echo "$dashboard_pid" >>"$out/pids"
for _ in $(seq 1 100); do
  grep -q "listening on " "$out/dashboard.log" 2>/dev/null && break
  sleep 0.1
done
dashboard_base=$(sed -n 's/.*listening on //p' "$out/dashboard.log" | head -1)
run_id=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().strip())['run']['id'])" "$out/export.jsonl")
curl -fsS "$dashboard_base/api/runs/$run_id" >"$out/dashboard.json"
kill "$dashboard_pid" 2>/dev/null || true

python3 "$here/fixture_provider.py" --mode invalid-duration --port-file "$scratch/invalid-port" 2>"$out/invalid-provider.log" &
echo $! >>"$out/pids"
for _ in $(seq 1 100); do
  [ -s "$scratch/invalid-port" ] && break
  sleep 0.1
done
invalid_port=$(cat "$scratch/invalid-port")
set +e
QIT_HOME="$state" "$binary" run core/speech-to-text-smoke \
  --provider local-cpu --model "$model" \
  --base-url "http://127.0.0.1:$invalid_port/$model" --iterations 1 --warmups 0 \
  >"$out/invalid.out" 2>"$out/invalid.err"
echo $? >"$out/invalid.status"
set -e

QIT_HOME="$state" "$binary" results --limit 5 >"$out/results-after-invalid.out" 2>&1

python3 "$here/assert_speech.py" "$root" "$out"
rm -rf "$scratch"
echo "evidence: $out"
