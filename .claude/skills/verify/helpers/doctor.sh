#!/bin/sh
# Read-only readiness check for the q.it verify skill.
set -eu
root=$(cd "$(dirname "$0")/../../../.." && pwd)
binary="$root/target/debug/qit"
status=0

if [ ! -x "$binary" ]; then
  echo "FAIL binary missing: $binary (run: cargo build --locked -p qit)"
  status=1
else
  echo "binary: $binary"
  "$binary" --version || status=1
  newest=$(find "$root/qit-runtime/src" "$root/qit/src" -name '*.rs' -newer "$binary" 2>/dev/null | head -1 || true)
  if [ -n "$newest" ]; then
    echo "FAIL binary is older than $newest (rebuild before driving)"
    status=1
  else
    echo "build: current"
  fi
fi

python=$(command -v python3 || true)
if [ -z "$python" ]; then
  echo "FAIL python3 is required by helpers/fixture_provider.py"
  status=1
else
  echo "python: $("$python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
fi

echo "revision: $(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$root" status --porcelain 2>/dev/null)" ]; then
  echo "checkout: dirty (evidence records the working tree, not a commit)"
else
  echo "checkout: clean"
fi
echo "evidence root: $root/.qit-evidence/verify"
exit "$status"
