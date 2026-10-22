#!/usr/bin/env bash
# Подъём БОТА НОВОСТЕЙ: хостовый процесс uvicorn на 127.0.0.1:${BRIDGE_NEWS_PORT:-8086}.
#
# Почему отдельный скрипт, а не флаг у основного моста: у основного жёстко зашиты свой порт, своя точка
# входа, свой pid-файл и свой health-URL — переиспользовать нельзя. Лог обязателен: без него причина
# смерти процесса не восстановима.
#
#   bash bridge-up-news.sh          # поднять в фоне (setsid+nohup, лог, pid-файл)
#   bash bridge-up-news.sh run      # запустить в ПЕРЕДНЕМ плане (для systemd-юнита)
#   bash bridge-up-news.sh status   # только проверка
#   bash bridge-up-news.sh stop     # остановить
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#: ❗ Своя библиотека подключается ЗДЕСЬ и до первой ссылки на корни: `$WORKSPACE` до неё не определён, а
#: `set -u` обрывает запуск на первой же такой ссылке. Хостовый лончер — путь ОТКАТА, и он обязан
#: запускаться.
source "$PROJECT_DIR/lib/workspace.sh"   #: PROJECT_DIR и WORKSPACE (с учётом `AGENT_WORKSPACE`)
source "$PROJECT_DIR/lib/vendor.sh"      #: режим платформы — один вход
source "$PROJECT_DIR/lib/dsn.sh"         #: хостовые DSN — один вход

ENV_FILE="${NEWS_ENV:-$PROJECT_DIR/.env}"                    #: окружение ПРОЕКТА
BASE_ENV="${NEWS_BASE_ENV:-$WORKSPACE/.env}"                 #: общая основа (может отсутствовать)
#: Интерпретатор — из своего окружения проекта, иначе из окружения воркспейса, иначе системный:
#: клон без венв тоже обязан запускаться, лишь бы в интерпретаторе стоял `news_agent`.
PYTHON="${NEWS_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "$PROJECT_DIR/.venv/bin/python" "$WORKSPACE/.venv/bin/python"; do
    [[ -x "$candidate" ]] && { PYTHON="$candidate"; break; }
  done
fi
PYTHON="${PYTHON:-python3}"
LOG="$PROJECT_DIR/logs/news.log"
PIDFILE="$PROJECT_DIR/news.pid"
PORT="${BRIDGE_NEWS_PORT:-8086}"
HEALTH="http://127.0.0.1:${PORT}/health"
ENTRY="news_agent.news_main:app"
PROC="uvicorn ${ENTRY}"
ACTION="${1:-up}"

health() { curl -fsS -m 3 "$HEALTH" 2>/dev/null; }

#: Хостовые адреса задаются ЯВНО и перекрывают `.env`: в нём адреса ВНУТРИ docker-сети
#: (`pgvector:5432`, `mcp-searxng:3000`, `host.docker.internal:8082`), а хостовый процесс таких имён не
#: разрешает. Нерезолвящийся адрес не роняет запуск — половина RAG отключается молча, поэтому
#: перекрываются ВСЕ адреса, а не только адрес базы.
#: ❗ Комментарии стоят НАД командой, а не внутри продолжения: строка `#` в продолжении с `\` разрывает
#: команду, и `env` остаётся без аргументов.

#: Режим читаем из того файла, где он лежит: общая основа, а если её нет — окружение проекта.
MODE_ENV="$BASE_ENV"
[[ -f "$MODE_ENV" ]] || MODE_ENV="$ENV_FILE"
#: Без этого режим читался бы из другого файла, чем тот, с которым работает служба.
require_platform_mode "бот новостей (хостовый запуск)" "$MODE_ENV"

case "$ACTION" in
  status)
    if out="$(health)"; then echo "[news] работает: $out"; else echo "[news] НЕ отвечает на $HEALTH"; fi
    ;;
  stop)
    pids="$(pgrep -f "$PROC" || true)"
    if [[ -z "$pids" ]]; then echo "[news] процесс не найден"; exit 0; fi
    kill $pids 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "[news] остановлен (pid: $pids)"
    ;;
  run)
    # Передний план: systemd сам следит за процессом (Restart=always) и пишет лог.
    mkdir -p "$PROJECT_DIR/logs"
    exec env PYTHONPATH="$PROJECT_DIR" \
      TEAM_MCP_SERVERS="${TEAM_MCP_SERVERS_HOST:-searxng=http://127.0.0.1:3001/mcp}" \
      DB_URL="$(dsn "$BASE_ENV" "$ENV_FILE")" \
      VECTOR_DB_URL="$(vector_dsn "$BASE_ENV" "$ENV_FILE")" \
      PLATFORM_API_URL="${PLATFORM_API_URL_HOST:-http://127.0.0.1:8680/api}" \
      LLAMA_EMBED_URL="${LLAMA_EMBED_URL_HOST:-http://127.0.0.1:8082/v1}" \
      SOLVER_URL="${SOLVER_URL_HOST:-http://127.0.0.1:8083}" \
      SESSIONS_DIR="${SESSIONS_DIR_HOST:-$PROJECT_DIR/.sessions}" \
      AGENT_ENV_FILES="$BASE_ENV:$ENV_FILE" \
      "$PYTHON" -m uvicorn "$ENTRY" --host 127.0.0.1 --port "$PORT"
    ;;
  up|*)
    if out="$(health)"; then echo "[news] уже поднят: $out"; exit 0; fi
    mkdir -p "$PROJECT_DIR/logs"
    setsid nohup env PYTHONPATH="$PROJECT_DIR" \
      TEAM_MCP_SERVERS="${TEAM_MCP_SERVERS_HOST:-searxng=http://127.0.0.1:3001/mcp}" \
      DB_URL="$(dsn "$BASE_ENV" "$ENV_FILE")" \
      VECTOR_DB_URL="$(vector_dsn "$BASE_ENV" "$ENV_FILE")" \
      PLATFORM_API_URL="${PLATFORM_API_URL_HOST:-http://127.0.0.1:8680/api}" \
      LLAMA_EMBED_URL="${LLAMA_EMBED_URL_HOST:-http://127.0.0.1:8082/v1}" \
      SOLVER_URL="${SOLVER_URL_HOST:-http://127.0.0.1:8083}" \
      SESSIONS_DIR="${SESSIONS_DIR_HOST:-$PROJECT_DIR/.sessions}" \
      AGENT_ENV_FILES="$BASE_ENV:$ENV_FILE" \
      "$PYTHON" -m uvicorn "$ENTRY" --host 127.0.0.1 --port "$PORT" \
      >> "$LOG" 2>&1 < /dev/null &
    pgrep -f "$PROC" > "$PIDFILE" 2>/dev/null || true
    for i in $(seq 1 20); do
      if out="$(health)"; then echo "[news] поднят (попытка $i): $out"; exit 0; fi
      sleep 2
    done
    echo "[news] НЕ поднялся за 40 с — хвост $LOG:"
    tail -n 25 "$LOG" || true
    exit 1
    ;;
esac
