"""Pytest-фикстуры. Сами фейки — в `news/tests/fakes.py` (их импортируют и тесты).

**`bridge_test_dsn` — своя база для тестов, которые проверяют SQL.**

Без неё проверки идут по БОЕВОЙ базе продукта и потому ничего не доказывают: тест очереди берёт
**чужую** строку `new` (самую старую из боевой базы), а прогон **пересоздаёт** таблицы чекпойнтов,
когда адрес по умолчанию смотрит на боевую базу (`…/vectors`).

**Как устроено.** Тестовая база создаётся на том же сервере, что боевая (значит DDL и драйвер ведут себя
одинаково), называется `news_test` и **пересоздаётся на каждый прогон**: `DROP … WITH (FORCE)` +
`CREATE`. Схему накатывает ТОТ ЖЕ код, что в бою — `Db(dsn, apply_ddl=True)`: копии DDL в тестах нет и
появиться не может. В конце сессии база удаляется.

**Сторож пригодности проверки** — `empty_test_db`: он утверждает, что в тестовой базе нет чужих строк.
Без него «тест зелёный» ничего не доказывал бы: если фикстуру однажды направят на боевую базу, проверки
вида «строка стала `sent`» пойдут по чужим данным.
"""
from __future__ import annotations

import asyncio
import os

import pytest

#: Адреса сервисов моделей объявляет АГЕНТ: тестам они тоже нужны и задаются в ОКРУЖЕНИИ до импорта
#: настроек — `Settings()` собирается на импорте модуля и без адреса откажет. Умолчаний в коде нет
#: намеренно: пустой адрес в бою должен быть виден, а не подменён «удобным».
for _name, _value in {
    "LLAMA_EMBED_URL": "http://embed.test:8082/v1",
    "LLAMA_EMBED_MODEL": "test-embed",
    "SOLVER_URL": "http://solver.test:8083",
}.items():
    os.environ.setdefault(_name, _value)

from fakes import db  # noqa: F401 — фикстура продукта
from live_db import assert_no_foreign_rows, env_value, host_dsn

TEST_DB_NAME = "news_test"


async def _recreate_database(admin_dsn: str, *, name: str) -> None:
    """Пересоздать тестовую базу: без этого тесты зависят от того, что осталось от прошлого прогона."""
    import asyncpg

    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def _drop_database(admin_dsn: str, *, name: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await conn.close()


async def _apply_schema(dsn: str) -> None:
    """Схему создаёт ТОТ ЖЕ код, что в бою (`apply_ddl=True`) — один источник DDL, никакого дубля."""
    from news_agent.db.repo import NewsDb

    repo = NewsDb(dsn, apply_ddl=True)
    await repo.connect()
    await repo.close()


@pytest.fixture(scope="session")
def bridge_test_dsn() -> str:
    """DSN тестовой базы: пересоздаётся на прогон, схема — как у боевой.

    Синхронная фикстура намеренно: сессионные асинхронные фикстуры требуют своего loop scope, а здесь
    достаточно `asyncio.run` — сам тест живёт в своей петле.
    """
    pytest.importorskip("asyncpg")
    url = env_value("DB_URL")
    if not url:
        pytest.skip("нет DB_URL в .env продукта — живые тесты не запускаются")
    server = host_dsn(url, db_name="postgres")   # служебная база того же сервера
    try:
        asyncio.run(_recreate_database(server, name=TEST_DB_NAME))
    except Exception as error:  # noqa: BLE001 — недоступный сервер это пропуск, а не отказ
        message = f"сервер базы недоступен ({type(error).__name__}: {error}) — проверьте docker ps"
        # ❗ Пропуск в прогоне, который зовут ИМЕННО ради живого пути, — это rc=0 без единой проверки,
        # то есть ложная зелень. Обычному гейту пропуск законен: поднятой базы он не требует.
        if os.environ.get("AGENT_LIVE_STRICT"):
            pytest.fail(message)
        pytest.skip(message)

    dsn = host_dsn(url, db_name=TEST_DB_NAME)
    asyncio.run(_apply_schema(dsn))
    yield dsn
    try:
        asyncio.run(_drop_database(server, name=TEST_DB_NAME))
    except Exception as error:  # noqa: BLE001 — ошибка удаления тестовой базы не влияет на исход прогона
        print(f"[conftest] тестовая база не удалена: {type(error).__name__}: {error}")


@pytest.fixture
async def empty_test_db(bridge_test_dsn: str) -> str:
    """Тестовая база пуста — иначе проверки пошли бы по чужим строкам (см. докстроку модуля)."""
    await assert_no_foreign_rows(bridge_test_dsn, where="тестовая база")
    yield bridge_test_dsn
