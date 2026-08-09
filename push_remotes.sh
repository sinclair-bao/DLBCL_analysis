#!/usr/bin/env bash
# Create empty remotes in the browser first, then run this script:
#   GitHub: https://github.com/new  →  sinclair-bao/DLBCL_analysis  (空仓库，不要 README)
#   Gitea:  http://100.100.211.88:3000/repo/create → sinclair/DLBCL_analysis
set -euo pipefail
cd "$(dirname "$0")"
git push -u origin main
git status -sb
echo "Pushed to GitHub + Gitea (origin dual-push)."
