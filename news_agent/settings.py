"""Настройки новостного контура: бот, каналы, лимиты, бюджет и цены ролей.

Общая часть живёт в `agent_core.config.CoreSettings`.
"""
from __future__ import annotations

from pydantic import Field

from agent_core.config import CoreSettings


class Settings(CoreSettings):
    # --- Адреса сервисов моделей (объявляет АГЕНТ, а не ядро) ---
    #: Службы моделей локальные, но адреса к ним — свойство развёртывания: в бою они могут стоять на
    #: разных хостах, поэтому адрес приходит ТОЛЬКО из `.env` этого агента. Пустое значение — отказ на
    #: старте с именем ключа: «удобное умолчание» означало бы запрос не туда и невидимую причину.
    llama_embed_url: str = Field(..., min_length=1)
    llama_embed_api_key: str = ""
    llama_embed_model: str = Field(..., min_length=1)
    llama_embed_timeout_s: float = 15.0
    solver_url: str = Field(..., min_length=1)
    solver_api_key: str = ""
    solver_timeout_s: float = 60.0
    telegram_news_bot_token: str = ""
    telegram_news_channel_id: str = ""        # рабочий канал публикации (-100…)
    telegram_news_test_channel_id: str = ""   # приватный тестовый канал для первой публикации
    telegram_news_owner_chat_id: str = ""     # пусто ⇒ берём telegram_owner_chat_id
    bridge_news_port: int = 8086
    news_topic: str = ""                      # пусто ⇒ дефолтная тема в news_bot
    news_max_items: int = 5
    news_max_per_day: int = 5                 # прогон ≈ $0.95: потолок держим по деньгам, не по времени
    news_month_budget_usd: float = 5.0        # месячный гейт расхода на новости (0 ⇒ без гейта)
    news_cooldown_s: int = 120
    news_stale_h: int = 12                    # старше — публикуем только с подтверждением
    news_edit_ttl_s: int = 600                # TTL режима правки (как owner_flow)
    news_publish_dry_run: bool = False
    # канал агента новостей: агент `news-hunter` с MCP-поиском searxng
    agent_bot_news: str = "news-hunter"
    news_connection_slug: str = "news-hunter-conn"
    news_workflow_id: str = ""
    news_revision_id: str = ""
    # Цены для пересчёта токенов в деньги. Фреймворк отдаёт ТОЛЬКО токены (`UsageMetrics` без
    # денег), а таблице сравнения нужна «стоимость прогона». Ноль означает «цена не задана»:
    # тогда стоимость команды в отчёте будет НЕИЗВЕСТНОЙ, а не нулевой — ноль выглядел бы как
    # бесплатная команда и занизил бы сравнение.
    team_price_in_per_1k_usd: float = 0.0
    team_price_out_per_1k_usd: float = 0.0
    # Источники инструментов по MCP: адреса через запятую, можно `имя=адрес`. Пусто ⇒
    # MCP-инструментов нет (команда работает со своими). Composio отклонён в пользу MCP
    # (см. `news_agent.news_team.tools.mcp`).
    team_mcp_servers: str = ""
    # Агент новостей и ревизии ролей: по ним трейс выпуска привязывается к агенту и к ролям.
    # Пустое значение = «объекта нет», и спан
    # уйдёт без этой ссылки (трассировка не должна зависеть от того, заведён ли объект).
    otel_agent_id: str = ""
    otel_role_researcher_revision: str = ""
    otel_role_author_revision: str = ""
    otel_role_verifier_revision: str = ""
    # Агент новостного выпуска: у выпуска СВОЙ объект (`news-hunter`), а `OTEL_WORKFLOW_*` описывают
    # агента диалога — ссылать выпуск на него значило бы приписать новости не тому агенту.
    otel_news_workflow_id: str = ""
    otel_news_workflow_slug: str = "news-hunter"

    @property
    def news_owner_chat_id(self) -> str:
        """Владелец новостного бота: свой чат, иначе общий (в приватном чате с ботом chat_id == user id)."""
        return self.telegram_news_owner_chat_id or self.telegram_owner_chat_id


settings = Settings()
