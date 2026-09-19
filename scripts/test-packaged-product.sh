#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/qit-package-smoke.XXXXXX")
qit_pid=""

cleanup() {
  if [[ -n "$qit_pid" ]]; then
    kill "$qit_pid" 2>/dev/null || true
    wait "$qit_pid" 2>/dev/null || true
  fi
  rm -rf "$smoke_dir"
}
trap cleanup EXIT

cd "$repo_dir"
cargo package -p qit-runtime --allow-dirty --no-verify
cargo package -p qit --allow-dirty --no-verify --list > "$smoke_dir/qit-package-files"

mkdir -p "$smoke_dir/source/runtime" "$smoke_dir/source/qit"
tar -xzf "$repo_dir/target/package/qit-runtime-0.1.0.crate" -C "$smoke_dir/source/runtime"
cp -R "$repo_dir/qit" "$smoke_dir/source/qit/qit-0.1.0"
cp "$repo_dir/Cargo.lock" "$smoke_dir/source/qit/qit-0.1.0/Cargo.lock"

runtime_path="$smoke_dir/source/runtime/qit-runtime-0.1.0"
ln -s "$runtime_path" "$smoke_dir/source/qit/qit-runtime"

CARGO_TARGET_DIR="$smoke_dir/target" cargo install \
  --path "$smoke_dir/source/qit/qit-0.1.0" \
  --root "$smoke_dir/install" \
  --locked \
  --offline

test -x "$smoke_dir/install/bin/qit"
test ! -e "$smoke_dir/install/bin/qit-stub-worker"

cd "$smoke_dir"
QIT_HOME="$smoke_dir/state" \
QIT_MODELS_DIR="$smoke_dir/models" \
QIT_PORT=0 \
"$smoke_dir/install/bin/qit" > "$smoke_dir/stdout" 2> "$smoke_dir/stderr" &
qit_pid=$!

base_url=""
for _ in {1..100}; do
  base_url=$(sed -n 's/^q.it listening on //p' "$smoke_dir/stdout")
  if [[ -n "$base_url" ]]; then
    break
  fi
  sleep 0.05
done
test -n "$base_url"

curl --fail --silent "$base_url/api/health" | grep --quiet '"ok":true'
curl --fail --silent "$base_url/" > "$smoke_dir/index.html"
asset_path=$(sed -n 's/.*src="\([^"]*\.js\)".*/\1/p' "$smoke_dir/index.html")
test -n "$asset_path"
curl --fail --silent "$base_url$asset_path" > "$smoke_dir/app.js"
test "$(wc -c < "$smoke_dir/app.js")" -gt 100000
