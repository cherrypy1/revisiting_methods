"""SSH from login node to the GPU node and start ``run_pag_only.py``.

Replaces ``dispatch_pag.sh``. Loads the gnu14 module, activates the venv,
then nohups the runner so it survives our SSH disconnect.

Usage:
    python scripts/dispatch_pag.py <compute_node>

Example:
    python scripts/dispatch_pag.py cn-006
"""

from __future__ import annotations

import argparse
import subprocess
import sys


REMOTE_LAUNCHER = (
    "source /etc/profile.d/modules.sh 2>/dev/null; "
    "module purge 2>/dev/null; "
    "module load gnu14/14.1 2>/dev/null; "
    "source ~/.venv/bin/activate; "
    "cd ~/geneval; "
    "nohup python scripts/run_pag_only.py > ~/pag_rerun.log 2>&1 &"
    " disown; "
    "sleep 2; "
    "pgrep -af run_pag_only"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("node", help="e.g. cn-006")
    args = p.parse_args()

    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        args.node,
        REMOTE_LAUNCHER,
    ]
    print(f"$ {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
