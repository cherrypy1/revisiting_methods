#!/usr/bin/env python3
"""
Wrapper script for create_prompts.py that ensures correct working directory.

Usage:
    python scripts/prompts/create_prompts.py --seed 42 -n 100 -o ./my_prompts

This wrapper changes to the prompts/ directory before running create_prompts.py
to ensure object_names.txt is found correctly.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    # GenEval prompts + create_prompts.py live in the external benchmark repo.
    project_root = Path(__file__).resolve().parents[2]
    geneval_root = Path(os.environ.get("GENEVAL_ROOT", str(project_root / "geneval-bench")))
    prompts_dir = geneval_root / "prompts"
    create_prompts_script = prompts_dir / "create_prompts.py"
    
    if not create_prompts_script.exists():
        print(f"Error: {create_prompts_script} not found", file=sys.stderr)
        sys.exit(1)
    
    # Change to prompts directory and run
    original_dir = os.getcwd()
    os.chdir(prompts_dir)
    
    try:
        cmd = [sys.executable, str(create_prompts_script)] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
