#!/usr/bin/env bash
# Хостовые DSN для процессов, запускаемых НЕ в контейнере.
#
# **Зачем отдельным файлом.** `.env` хранит адреса ВНУТРИ docker-сети (`data-db:5432`, `pgvector:5432`),
# а хостовый процесс таких имён не разрешает. Перекрытие на опубликованные порты (55435 и 55433) нужно
# нескольким процессам, поэтому живёт в одном месте: копия разойдётся, и отказ будет тихим — бот
# поднимется без базы знаний, процесс без чекпойнтера, и оба скажут, что «механизм не поднят».
#
# Подключение:  source "$PROJECT_DIR/lib/dsn.sh"

#: Креды и имя базы — из файлов окружения, хост и порт — опубликованные на 127.0.0.1.
#: Файлы читаются ПО ПОРЯДКУ: побеждает последний, где ключ определён. Имя базы берётся из пути `DB_URL`
#: (своя база проекта), поэтому лончер передаёт оба файла: общую основу и окружение проекта.
dsn() {
  python3 - "$@" <<'DSN_PY'
import os, sys, urllib.parse as u

raw = {}
for path in sys.argv[1:] or [".env"]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    raw[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        continue

url = raw.get("DB_URL") or raw.get("BRIDGE_DB_URL")
if not url:
    raise SystemExit("[dsn] нет DB_URL ни в одном из файлов окружения — база продукта не задана")
p = u.urlsplit(url)
cred = p.netloc.rsplit("@", 1)[0] + "@" if "@" in p.netloc else ""
db = p.path.lstrip("/")
if not db:
    raise SystemExit(f"[dsn] в DB_URL ({p.hostname}) не названа база — имя базы задаётся адресом")
#: ❗ Порт — по имени хоста ИЗ DSN, а не литералом: «единый источник» не должен врать первым.
#: Неизвестный хост — ОТКАЗ с причиной: тихое «5432» уже давало молчаливый отказ подключения.
PORTS = {"realty-db": 55435, "news-db": 55436, "pgvector": 55433}
port = os.environ.get("BRIDGE_DB_PORT_HOST") or PORTS.get(p.hostname or "")
if port is None:
    raise SystemExit(f"[dsn] хост базы {p.hostname!r} из DB_URL неизвестен: "
                     f"опубликованные порты знаю для {sorted(PORTS)} (BRIDGE_DB_PORT_HOST — переопределение)")
print(f"postgresql+asyncpg://{cred}127.0.0.1:{port}/{db}")
DSN_PY
}

vector_dsn() {
  python3 - "$@" <<'DSN_PY'
import os, sys, urllib.parse as u
raw = {}
for path in sys.argv[1:] or [".env"]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    raw[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        continue
url = raw.get("VECTOR_DB_URL", "")
if not url:
    raise SystemExit("[dsn] нет VECTOR_DB_URL в файлах окружения — адрес векторной базы задаётся ЯВНО")
p = u.urlsplit(url)
cred = p.netloc.rsplit("@", 1)[0] + "@" if "@" in p.netloc else ""
db = p.path.lstrip("/") or "vectors"
port = os.environ.get("BRIDGE_VECTOR_PORT_HOST") or 55433
print(f"postgresql+asyncpg://{cred}127.0.0.1:{port}/{db}")
DSN_PY
}
