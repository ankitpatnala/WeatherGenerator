#!/bin/bash
# Generate eval-style decadal animations for several fields (sequential).
#cd "$(dirname "$0")/.."
source .venv/bin/activate
FIELDS="${FIELDS:-t_850 z_500 z_850 u_850 v_850 q_850 2t}"
for f in $FIELDS; do
  echo "===== $(date +%H:%M) generating $f ====="
  .venv/bin/python3 diagnostics/animate_decadal.py --field "$f" --fps 8 || echo "FAILED: $f"
done
echo "===== $(date +%H:%M) all done ====="
