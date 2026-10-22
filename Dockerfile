# Образ НОВОСТНОГО контура (`news-agent`): свой проект, свой пакет (`news_agent`), общий слой — колёсом.
#
# Multi-stage: колёса собираются отдельно от рантайма (меньше образ, нет build-зависимостей в проде).
FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /wheels
COPY pyproject.toml ./
# Колесо ОБЩЕГО СЛОЯ (`agent-core`) лежит в репозитории готовым — `wheels/` — и берётся оттуда: этого
# пакета нет в индексе, и без локального файла разрешение `agent-core==…` ушло бы в сеть за тем, чего
# там нет. Этап сборки колёс сеть использует (pip, build, setuptools, зависимости проекта); этап
# рантайма ставит всё с `--no-index` — только из `/wheels` и `/wheels-core`.
COPY wheels /wheels-core
# `setuptools` кладём в /wheels рядом с зависимостями: установка ниже идёт с `--no-index`, и бэкенд
# сборки должен лежать локально (иначе `Could not find a version that satisfies the requirement setuptools`).
RUN pip install --upgrade pip build && pip wheel --wheel-dir /wheels --find-links=/wheels-core . setuptools

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2

#: Рабочий каталог проекта. ❗ Каталог файлов-сессий НЕ выводится из расположения кода (общий слой ставится
#: колёсом и живёт в site-packages): он приходит настройкой `SESSIONS_DIR`, и её задаёт композ — иначе
#: получается путь внутри site-packages, куда писать нельзя, и бот молча не подключается.
WORKDIR /srv/news
COPY --from=builder /wheels /wheels
COPY wheels /wheels-core
COPY pyproject.toml ./
COPY news_agent ./news_agent
RUN pip install --no-cache-dir --no-index --find-links=/wheels --find-links=/wheels-core setuptools \
 && pip install --no-cache-dir --no-index --find-links=/wheels --find-links=/wheels-core --no-build-isolation -e . \
 && rm -rf /wheels /wheels-core

EXPOSE 8086
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8086/health', timeout=5).status_code==200 else 1)"

# Один процесс — один event loop: у новостей свой граф и своя очередь выпусков.
CMD ["uvicorn", "news_agent.news_main:app", "--host", "0.0.0.0", "--port", "8086", "--no-access-log"]
