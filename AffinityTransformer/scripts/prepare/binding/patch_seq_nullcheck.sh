#!/usr/bin/env bash
# One-time migration: fix _seq() null-check in all binding convert.py files.
#
# Problem: `s in ("NAN","NONE","NA","")` does whole-string comparison, so a
# real 2-AA sequence "NA" (Asn-Ala) or 3-AA "NAN" would be incorrectly dropped.
#
# Fix: check the original value's type before str() conversion.
#   pandas missing values arrive as float('nan'), caught by math.isfinite().
#   The string "NA" / "NAN" is then treated as a valid AA fragment (correct).
#
# Usage (from repo root):
#   bash scripts/prepare/binding/patch_seq_nullcheck.sh
#
# Safe to re-run: skips files already patched (no old pattern found).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python3 - "$ROOT" << 'EOF'
import sys, re
from pathlib import Path

root = Path(sys.argv[1])

# Matches the two-line block regardless of exact quote style / spacing:
#   s = str(val).strip().upper()
#   if not s or s in ("NAN", "NONE", ...): return None
OLD_BLOCK = re.compile(
    r'    s = str\(val\)\.strip\(\)\.upper\(\)\n'
    r'    if not s or s in \([^)]+\): return None\n',
    re.MULTILINE,
)

NEW_BLOCK = (
    '    if isinstance(val, float) and not math.isfinite(val): return None\n'
    '    s = str(val).strip().upper()\n'
    '    if not s: return None\n'
)

count = 0
for p in sorted(root.glob("scripts/prepare/binding/**/convert.py")):
    text = p.read_text()
    if OLD_BLOCK.search(text):
        p.write_text(OLD_BLOCK.sub(NEW_BLOCK, text))
        print(f"  patched  {p.relative_to(root)}")
        count += 1

print(f"\nDone: patched {count} file(s)")
EOF
