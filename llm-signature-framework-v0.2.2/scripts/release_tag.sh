#!/usr/bin/env bash
set -euo pipefail
VERSION="0.2.2"
git add -A
git commit -m "chore(release): v$VERSION"
git tag -a "v$VERSION" -m "release: v$VERSION"
echo "Tagged v$VERSION. Push with: git push origin main --tags"
