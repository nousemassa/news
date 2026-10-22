#!/usr/bin/env bash
# Проект `news`: бот канала новостей и его процесс. Своя папка, свой композ, своё имя проекта compose
# (`news`), свой каталог сессий.
#
#   bash news-up.sh            # поднять службу и дождаться готовности
#   bash news-up.sh status     # показать состояние (включая Telegram и канал)
#   bash news-up.sh stop       # остановить
#
# **Что этот лончер обязан задать ЯВНО.** Обязательно только окружение ПРОЕКТА (`$PROJECT_DIR/.env`);
# общий `.env` воркспейса подключается, если он есть, — проект может стоять и отдельно. Лончер печатает,
# ОТКУДА берёт окружение, и проверяет, что в нём есть каталог сессий: неверный путь означает, что бот
# молча не подключится.
set -euo pipefail

#: Корни проекта (`PROJECT_DIR`/`WORKSPACE`) даёт своя библиотека: она же разрешает `AGENT_WORKSPACE`,
#: когда проект стоит отдельно от воркспейса.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/lib/workspace.sh"
#: Режим платформы — ОДИН вход (`lib/vendor.sh`): в чужом режиме продукт поднимать не на чем, и лончер
#: обязан сказать это ПРИЧИНОЙ, а не падением миграции.
source "$PROJECT_DIR/lib/vendor.sh"

COMPOSE="$PROJECT_DIR/compose.news.yaml"
ENV_FILE="${NEWS_ENV:-$PROJECT_DIR/.env}"                        #: окружение ПРОЕКТА (обязательно)
BASE_ENV="${NEWS_BASE_ENV:-$WORKSPACE/.env}"                     #: общая основа (необязательна)

[[ -f "$ENV_FILE" ]] || { echo "[news] ОСТАНОВ: нет окружения проекта $ENV_FILE" >&2; exit 2; }
echo "[news] окружение: $ENV_FILE"
if [[ -f "$BASE_ENV" ]]; then
  echo "[news] общая основа: $BASE_ENV"
else
  echo "[news] общей основы нет ($BASE_ENV) — работаю без неё"
fi

#: Аргументы композа — СПИСКОМ: необязательный файл попадает в него, только если он есть.
#: Порядок важен: побеждает последний файл, поэтому окружение проекта идёт после общей основы.
COMPOSE_ENV_ARGS=(--env-file "$ENV_FILE")
if [[ -f "$BASE_ENV" ]]; then
  COMPOSE_ENV_ARGS=(--env-file "$BASE_ENV" "${COMPOSE_ENV_ARGS[@]}")
fi
compose() { docker compose "${COMPOSE_ENV_ARGS[@]}" -f "$COMPOSE" "$@"; }

#: Каталог сессий проверяем ДО старта: файл-сессия — ключ доступа бота, и без каталога он не подключится.
[[ -d "$PROJECT_DIR/.sessions" ]] || {
  echo "[news] ОСТАНОВ: нет каталога $PROJECT_DIR/.sessions — бот не сможет хранить сессию" >&2; exit 2; }

ACTION="${1:-up}"

status() {
  local out
  if ! out="$(curl -fsS -m 10 "http://127.0.0.1:8086/health" 2>/dev/null)"; then
    echo "[news] НЕ отвечает на 127.0.0.1:8086 ✗"; return 1
  fi
  python3 - "$out" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
print(f"[news] /health: telegram={d.get('telegram')} channel={d.get('channel')} "
      f"режим={d.get('transport')} выпусков={d.get('runs')}")
PY
  python3 -c "
import json,sys
d=json.loads(sys.argv[1])
sys.exit(0 if d.get('telegram') and d.get('channel') else 1)" "$out"
}

case "$ACTION" in
  status) status ;;
  stop)
    compose stop news
    echo "[news] остановлено (тома и данные не тронуты)."
    ;;
  up|*)
    #: Режим читаем из того файла, где он лежит: общая основа, а если её нет — окружение проекта.
    MODE_ENV="$BASE_ENV"
    [[ -f "$MODE_ENV" ]] || MODE_ENV="$ENV_FILE"
    require_platform_mode "бот новостей" "$MODE_ENV"
    compose up -d news
    for i in $(seq 1 40); do status >/dev/null 2>&1 && { status; exit 0; }; sleep 3; done
    echo "[news] ОСТАНОВ: не подключился за 120 с — смотрите docker logs news" >&2
    exit 1
    ;;
esac
