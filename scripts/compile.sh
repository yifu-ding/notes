#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# 1. Make the repository's content directory match the Obsidian Vault.
npm run publish-content

# 2. Ensure packages declared by quartz.config.yaml are available, then verify
#    that the synced content builds successfully.
node ./quartz/bootstrap-cli.mjs plugin install --from-config
node ./quartz/bootstrap-cli.mjs build

# 3. Commit, pull remote changes, and push the v5 branch. Extra Quartz sync
#    flags can be forwarded, e.g. ./scripts/compile.sh --no-pull on first push.
node ./quartz/bootstrap-cli.mjs sync "$@"
