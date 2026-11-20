#!/usr/bin/env bash
# Recompute every result from the data in this package, in order, then compare the results with the
# published ones in published/ (verify.py).  Run from the package root:  bash reproduce.sh
# Programs that read downloaded workbooks are run with "python3 -I".
set -euo pipefail
step() { echo; echo "== $(date +%H:%M:%S)  $*"; }

step "backtest";                 python3 research/trend_backtest.py --intervals > /dev/null
step "trend baseline";           python3 research/trend_baseline.py > /dev/null
step "Taiwan 2025";              python3 -I research/taiwan_2025.py > /dev/null
step "age-group death rates";    python3 research/trend_age_rates.py > /dev/null
step "other baselines";          python3 research/trend_cohort_check.py > /dev/null
step "replication";              python3 research/trend_replicate_dowd.py > /dev/null
step "registration check";       python3 -I research/taiwan_registration_check.py > /dev/null
step "causes: prepare";          python3 -I research/cause_prepare.py > /dev/null
step "causes: decomposition";    python3 research/cause_decomposition.py > /dev/null
step "causes: ICD codes";        python3 -I research/cause_codes.py > /dev/null
step "context indicators";       python3 -I research/cause_context.py > /dev/null
step "tables, numbers, figures"; python3 research/paper_assets.py
step "comparison with the published results"; python3 verify.py
step "done"
