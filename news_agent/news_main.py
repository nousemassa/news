"""FastAPI-приложение бота новостей: ОТДЕЛЬНЫЙ процесс на `BRIDGE_NEWS_PORT` (по умолчанию 8086).

Схему НЕ создаёт (`apply_ddl=False`): единственный писатель — основной мост, новостной процесс
только проверяет наличие своих таблиц. Ручки: `GET /health` (для сторожа) и `POST /api/ops/news`
(ручной прогон для скриптов; создаёт ТОЛЬКО черновики — публиковать в канал ручкой нельзя).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from agent_core import logging_redact, startup, telemetry
from agent_core.api_auth import check_api_key
from agent_core.decisions import ladder
from agent_core.decisions.client import make_ask
from agent_core.vendors.runtime import runtime_for
from news_agent.agent_roles import news_role
from news_agent.db.repo import NEWS_TABLES, REVISION, NewsDb
from news_agent.settings import settings
from news_agent.tg.news_bot import NewsBot

log = logging.getLogger(__name__)

# Как в основном мосте: свои INFO-логи без этого не попадают в stderr; URL-содержащие логи глушим.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging_redact.install()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = NewsDb(
        settings.db_url, tz=settings.tz, apply_ddl=False,
        expected_tables=NEWS_TABLES,
        expected_component="news",  # проверяем именно свои таблицы, схему создаёт накат
        expected_revision=REVISION,
    )
    await startup.wait_ready(db.connect, what="БД (схема новостей)")

    #: Трассировка собирается на СТАРТЕ, а не при первом выпуске: провайдер, созданный лениво, оставлял
    #: бы HTTP-спаны (в т.ч. полезные — `/api/ops/news`) без приёмника, пока не случится первый прогон.
    #: Роль одна (`agent-news`) — этот процесс и есть агент новостей.
    telemetry.register_roles("news")
    app.state.tracer = telemetry.setup_telemetry(settings, agent="news")

    #: ❗ Через ПОРТ рантайма: делегат над клиентом (поведение не меняется).
    role = news_role()
    channels = runtime_for(settings, role=role)
    if role.workflow_id and role.revision_id:
        await startup.wait_ready(
            lambda: channels.ensure_connection(slug=role.connection_slug),
            what="канал агента новостей (news.ensure_connection)",
        )
        await startup.wait_ready(
            lambda: channels.ensure_agent(
                workflow_id=role.workflow_id,
                workflow_revision_id=role.revision_id,
                slug=role.bot,
            ),
            what="агент новостей (news.ensure_agent)",
        )

    # Контур команды ролей: поиск по базе знаний и гейт новости — боевой путь
    # новостей. База знаний необязательна: без неё исследователь идёт с одним внешним поиском, и
    # подъём процесса от этого не зависит (тот же приём, что у процесса команды).
    retriever = None
    from agent_core.db.store import Db

    vector_db = Db(settings.vector_db_url, apply_ddl=False)
    try:
        await startup.wait_ready(vector_db.connect, what="БД базы знаний", budget_s=15.0)
        from agent_core.retrieval.pgvector import PgVectorRetriever

        retriever = PgVectorRetriever(
            vector_db.pool,
            table="knowledge_chunks",
            fts_language="russian",
            embed_url=settings.llama_embed_url,
            embed_api_key=settings.llama_embed_api_key,
            embed_model=settings.llama_embed_model,
            embed_timeout_s=settings.llama_embed_timeout_s,
        )
    except Exception as error:  # noqa: BLE001 — без базы знаний бот обязан подниматься
        log.warning("news: база знаний недоступна (%s: %s) — исследователь пойдёт без неё",
                    type(error).__name__, error)
        vector_db = None

    bot = NewsBot(settings=settings, db=db, channels=channels, retriever=retriever,
                  gate_ask=make_ask(settings.solver_url, api_key=settings.solver_api_key,
                                timeout_s=settings.solver_timeout_s), gate_threshold=ladder.threshold_from(settings))
    await bot.start()

    app.state.settings = settings
    app.state.db = db
    app.state.news = bot
    # ❗ `try/finally`: при исключении в генератор (падение на старте/остановке) пулы и клиент канала
    # обязаны закрыться — закрытие только на «нормальном» пути оставляло их висеть.
    try:
        yield
    finally:
        await bot.stop()
        await channels.aclose()
        await db.close()
        if vector_db is not None:
            await vector_db.close()


app = FastAPI(title="news", lifespan=lifespan,
              telemetry=telemetry.fastapi_telemetry())


@app.get("/health")
async def health() -> dict:
    bot = getattr(app.state, "news", None)
    return bot.health() if bot else {"status": "starting"}


class NewsRunBody(BaseModel):
    query: str = ""          # тема прогона; пусто ⇒ дефолтная из NEWS_TOPIC/настроек


@app.post("/api/ops/news")
async def news_run(
    body: NewsRunBody,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    """Ручной прогон: создаёт черновики и присылает карточки овнеру. Публикация — только кнопкой в боте."""
    if not check_api_key(x_api_key, settings.bridge_api_key):
        raise HTTPException(status_code=401, detail="invalid X-API-Key")
    bot = getattr(request.app.state, "news", None)
    if bot is None or not bot.running:
        raise HTTPException(status_code=503, detail="news bot is not running")
    owner = settings.news_owner_chat_id
    if not owner:
        raise HTTPException(status_code=503, detail="не задан чат владельца (TELEGRAM_NEWS_OWNER_CHAT_ID)")
    # ❗ Через лок чата, как кнопка: иначе ручной прогон сторожа и нажатие кнопки шли бы одновременно.
    return {"status": "ok", **await bot.run_once_locked(chat_id=owner, topic=body.query or None)}
