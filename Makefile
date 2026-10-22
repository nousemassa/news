# Проект `news`: бот канала и его процесс. Свой Makefile — чтобы проект можно было поднять и проверить
# без воркспейса. Логика живёт в лончере, цели её не дублируют.
#: Интерпретатор с окружением проекта.
PY ?= python3
.PHONY: up status stop restart images check test ci

up:
	bash news-up.sh up

status:
	bash news-up.sh status

stop:
	bash news-up.sh stop

restart: stop up

#: Образ проекта — `news-agent:latest`: из него поднимаются и бот, и накат схемы. Пересборка заявлена
#: отдельной целью: `up` на уже существующем образе ничего не пересобирает.
images:
	docker compose --env-file .env -f compose.news.yaml build news

# Офлайн-проверка: композ обязан РАЗБИРАТЬСЯ настоящим парсером (демон не нужен).
#: `PLATFORM_NETWORK` обязателен (`:?` в композе), а для разбора нужна только подстановка — сеть при
#: этом не создаётся. Образец окружения берём из `.env.example`: проверка не требует настоящего `.env`.
check:
	@PLATFORM_NETWORK=__check__ docker compose --env-file .env.example -f compose.news.yaml config -q \
	  && echo "[news] композ разбирается ✓"

#: Офлайн-тесты. Живые (нужна поднятая база из `.env`) пропускают себя сами; чтобы пропуск стал
#: отказом — `AGENT_LIVE_STRICT=1 make test`.
test:
	$(PY) -m pytest tests -q

#: То же, что гоняет CI: линтер по своим файлам и свои офлайн-тесты.
ci:
	$(PY) -m ruff check .
	$(PY) -m pytest tests -q
