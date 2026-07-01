#!/usr/bin/env bash
# 用法（在任意目录执行其一）:
#   source ~/Desktop/Agent_SDR/sdr_env.sh
#   . ~/Desktop/Agent_SDR/sdr_env.sh
# 可在 ~/.bashrc 里加一行:  alias sdr='source ~/Desktop/Agent_SDR/sdr_env.sh'

_SDR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_SDR_ROOT" || return 1
# shellcheck disable=SC1091
source "$_SDR_ROOT/.venv/bin/activate"
unset _SDR_ROOT
