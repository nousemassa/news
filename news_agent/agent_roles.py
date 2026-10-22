"""Роль агента новостей: канал, ревизия и свойства для рантайма.

Свой рантайм для этой роли не реализован: у новостного контура другой харнесс (граф ролей), поэтому
`own_supported=False` — роль исполняет платформа, и это названный отказ, а не тихая подмена.
"""
from __future__ import annotations

from agent_core.vendors.runtime.port import RoleSpec
from news_agent.settings import settings


def news_role() -> RoleSpec:
    return RoleSpec(
        name="news",
        bot=settings.agent_bot_news,
        connection_slug=settings.news_connection_slug,
        connection_name="News Hunter",
        workflow_id=settings.news_workflow_id,
        revision_id=settings.news_revision_id,
        own_supported=False,
    )
