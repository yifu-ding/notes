#!/usr/bin/env bash

set -euo pipefail

# Obsidian is the source of truth. Override CONTENT_SOURCE to publish a
# different vault location when needed.
source_dir="${CONTENT_SOURCE:-/Users/yifuding/Documents/vault/实习面试准备/八股基础知识/chapters}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${repo_root}/content"

if [[ ! -d "${source_dir}" ]]; then
  printf 'Content source does not exist: %s\n' "${source_dir}" >&2
  exit 1
fi

if [[ -L "${target_dir}" ]]; then
  printf 'Refusing to sync into symbolic link: %s\n' "${target_dir}" >&2
  exit 1
fi

mkdir -p "${target_dir}"
rsync -a --delete --exclude='.DS_Store' "${source_dir}/" "${target_dir}/"

printf 'Synced Obsidian content to %s\n' "${target_dir}"
