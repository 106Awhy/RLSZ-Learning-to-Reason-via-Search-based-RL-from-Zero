#!/usr/bin/env bash
set -euo pipefail

ROOT="${SEARCH_R1_ROOT:-$PWD/.work}"
REPO="${SEARCH_R1_REPO:-$ROOT/repos/Search-R1}"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Missing Search-R1 repo at $REPO" >&2
  exit 2
fi

cd "$REPO"
if ! grep -q 'SEARCH_R1_REWARD' verl/trainer/main_ppo.py; then
  patch -p1 < "$PKG_DIR/patches/search_r1_reward_custom.patch"
fi

mkdir -p "$REPO/verl/utils/reward_score"
cp "$PKG_DIR/src/verl/utils/reward_score/qa_custom.py" \
  "$REPO/verl/utils/reward_score/qa_custom.py"

echo "Reward patch is installed in $REPO"
