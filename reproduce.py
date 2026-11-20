#!/usr/bin/env python3
"""Recompute every result from the data in this package, in order, then compare the results with the
ones in published/ (verify.py).

Run it with the Python that has the packages of requirements.txt installed, from any directory:

    python3 reproduce.py

Every program is started with that same Python, so no shell is needed and the script runs wherever
Python runs.  Programs that read downloaded workbooks are started in isolated mode (-I).  The script
stops at the first step that fails.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (what the step does, program, arguments, isolated mode, show the program's own output)
STEPS = [
    ('backtest', 'research/trend_backtest.py', ['--intervals'], False, False),
    ('trend baseline', 'research/trend_baseline.py', [], False, False),
    ('Taiwan 2025', 'research/taiwan_2025.py', [], True, False),
    ('age-group death rates', 'research/trend_age_rates.py', [], False, False),
    ('other baselines', 'research/trend_cohort_check.py', [], False, False),
    ('replication', 'research/trend_replicate_dowd.py', [], False, False),
    ('registration check', 'research/taiwan_registration_check.py', [], True, False),
    ('causes: prepare', 'research/cause_prepare.py', [], True, False),
    ('causes: decomposition', 'research/cause_decomposition.py', [], False, False),
    ('causes: ICD codes', 'research/cause_codes.py', [], True, False),
    ('context indicators', 'research/cause_context.py', [], True, False),
    ('tables, numbers, figures', 'research/paper_assets.py', [], False, True),
    ('comparison with the published results', 'verify.py', [], False, True),
]


def run(steps, root: Path) -> None:
    """Run the steps in order from `root`; stop with a message at the first one that fails."""
    for label, program, arguments, isolated, show in steps:
        print(f"\n== {time.strftime('%H:%M:%S')}  {label}", flush=True)
        command = [sys.executable] + (["-I"] if isolated else []) + [program] + arguments
        done = subprocess.run(command, cwd=root, stdout=None if show else subprocess.DEVNULL)
        if done.returncode != 0:
            sys.exit(f"the step '{label}' ({program}) ended with exit status {done.returncode}; nothing after it was run")
    print(f"\n== {time.strftime('%H:%M:%S')}  done", flush=True)


if __name__ == "__main__":
    run(STEPS, ROOT)
