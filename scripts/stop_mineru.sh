#!/usr/bin/env bash
set -euo pipefail

# 停止 start_mineru.sh 启动的 VLM 服务，并清理残留的 MinerU 文档库服务（本项目不使用它）。
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.mineru.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少 ${ENV_FILE}：无法确定 MinerU 的工作目录。" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a
PYTHON_ENV="${MINERU_PYTHON_ENV:-/opt/anaconda3/envs/langchain_env}"

# 1. VLM 服务：start_mineru.sh 把进程号写在 $MINERU_HOME/vlm-server.pid
PID_FILE="$MINERU_HOME/vlm-server.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  vlm_pid="$(cat "$PID_FILE")"
  kill "$vlm_pid"
  rm -f "$PID_FILE"
  echo "已停止 VLM 服务（PID ${vlm_pid}）"
else
  rm -f "$PID_FILE"
  echo "VLM 服务没有在运行（或不是由 start_mineru.sh 启动的）"
fi

# 2. 文档库服务：先正常停止；进程卡住不响应时 mineru server stop 会失败，
#    这时按启动命令行（python -m mineru.doclib.app）找到进程直接结束
if [[ -x "$PYTHON_ENV/bin/mineru" ]] && "$PYTHON_ENV/bin/mineru" server stop >/dev/null 2>&1; then
  echo "已停止 MinerU 文档库服务"
else
  # 只匹配"以 python 开头、带 -m mineru.doclib.app"的命令行，避免误杀碰巧包含这串文字的其他进程
  pids="$(pgrep -f '^[^ ]*python[0-9.]* -m mineru[.]doclib[.]app' || true)"
  if [[ -n "$pids" ]]; then
    echo "文档库服务没有响应，直接结束进程：${pids//$'\n'/ }"
    kill $pids
  else
    echo "MinerU 文档库服务没有在运行"
  fi
fi
