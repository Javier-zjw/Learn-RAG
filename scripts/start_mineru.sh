#!/usr/bin/env bash
set -euo pipefail

# 启动本机 MinerU 4.x 的 VLM 服务（llama.cpp）。
# 环境变量只放在 .mineru.env，不读取项目 .env；解析器（learn_rag/parsing/external.py）读取的是同一个文件。
# 本项目通过 mineru-kit parse 解析文档，不需要 MinerU 的文档库服务（mineru server），所以这里不启动它。
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.mineru.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少 ${ENV_FILE}：请先创建本机 MinerU 环境文件。" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

# MINERU_PYTHON_ENV 建议写在 .mineru.env 里；文件里没写时，也可以在命令行前面临时指定
PYTHON_ENV="${MINERU_PYTHON_ENV:-/opt/anaconda3/envs/langchain_env}"
MINERU_KIT="$PYTHON_ENV/bin/mineru-kit"

if [[ ! -x "$MINERU_KIT" ]]; then
  echo "MinerU 未安装在 ${PYTHON_ENV}，请先执行：${PYTHON_ENV}/bin/python3.12 -m pip install 'mineru>=4.0,<5'" >&2
  exit 1
fi
if [[ ! -d "$MINERU_HOME" ]]; then
  echo "模型目录不存在：$MINERU_HOME" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "启动脚本需要 curl 检查本机服务健康状态。" >&2
  exit 1
fi

VLM_HEALTH_URL="${MINERU_MODEL_VLM_SERVER_URL%/}/v1/models"
if curl --fail --silent --show-error "$VLM_HEALTH_URL" >/dev/null 2>&1; then
  echo "MinerU VLM 服务已运行：$MINERU_MODEL_VLM_SERVER_URL"
else
  mkdir -p "$MINERU_HOME/logs"
  nohup "$MINERU_KIT" vlm-server --engine llama-cpp -- \
    --host "$MINERU_VLM_HOST" --port "$MINERU_VLM_PORT" \
    >>"$MINERU_HOME/logs/vlm-server.log" 2>&1 &
  vlm_pid=$!
  echo "$vlm_pid" >"$MINERU_HOME/vlm-server.pid"
  echo "正在启动 CPU VLM 服务（PID ${vlm_pid}）..."

  ready=false
  for _ in $(seq 1 120); do
    if curl --fail --silent --show-error "$VLM_HEALTH_URL" >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "$vlm_pid" 2>/dev/null; then
      echo "VLM 服务启动失败，最近日志：" >&2
      tail -50 "$MINERU_HOME/logs/vlm-server.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo "VLM 服务 120 秒内未就绪，请查看 $MINERU_HOME/logs/vlm-server.log" >&2
    exit 1
  fi
  echo "MinerU VLM 服务已就绪：$MINERU_MODEL_VLM_SERVER_URL"
fi

echo "MinerU 已就绪：MINERU_HOME=${MINERU_HOME}"
