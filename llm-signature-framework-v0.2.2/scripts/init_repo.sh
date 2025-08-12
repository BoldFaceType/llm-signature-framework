#!/usr/bin/env bash
set -euo pipefail
git init
git add .
git commit -m "feat(core): initial vetting-ready framework v0.2.2 with CI (ruff+pytest+coverage gate)"
git branch -M main
git remote add origin <REMOTE>
git push -u origin main
