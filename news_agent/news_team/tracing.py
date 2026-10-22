"""Форма выпуска в трейсе: корень и роли-агента.

**Задача.** Агенты и пайплайн их взаимодействия должны быть **явно видны** на платформе.
Значит трейс выпуска должен читаться как дерево: корень-выпуск и дети-агенты (исследователь, автор,
проверяющий), с порядком передачи работы и с падением на **своём** ребре.

**Что здесь есть, а что приходит извне.** Здесь — то, что принадлежит ВЫПУСКУ НОВОСТЕЙ:
какие ссылки на объекты платформы берутся из настроек (`TraceRefs`, `trace_refs`) и что корень выпуска
считается агентом `news-hunter`. Самой формы атрибутов здесь нет: её собирает адаптер платформы
(`agent_core.vendors`) из НЕЙТРАЛЬНОГО описания спана. Одно место на все вендоры — иначе копия
объяснения разойдётся с кодом.

**Где измеренная форма и почему именно такая.** В адаптере платформы, рядом с пробой, которой
она снята: тип спана читается приёмом
только из ВЛОЖЕННОГО атрибута, ссылки работают плоскими ключами.

**Чего здесь нет.** Модуль не знает ни про провайдера, ни про Telegram: он получает трейсер и ссылки.
Общая часть — открытие спана, пометка ошибки, атрибуты узла и корня — живёт в `agent_core.spans` (её
делит с нами граф напоминаний). Когда трассировка выключена, `tracer` — пустышка (`ProxyTracer`), и все
контексты работают как «ничего не делать»: ни одна точка вызова не проверяет «включено ли».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent_core import vendors
from agent_core.spans import (
    mark_error,
    node_attributes,
    session_id_for,
    span,
)

# Реэкспорт: контур берёт эти помощники отсюда (одно место импорта для `news_team/graph.py`).
__all__ = [
    "NON_ROLE_NODES",
    "RELEASE_SPAN",
    "TraceRefs",
    "mark_error",
    "node_attributes",
    "release_attributes",
    "role_attributes",
    "span",
    "trace_refs",
]

log = logging.getLogger(__name__)

#: Имя выпуска в трейсе. Префикс `news.` — чтобы SQL-проверка и человек видели наши спаны отдельно.
RELEASE_SPAN = "news.release"

#: Узлы, которые НЕ являются ролями: их спаны идут типом `task` и без агента, иначе «три агента»
#: читались бы как шесть.
NON_ROLE_NODES = ("prepare", "gate", "assemble")


@dataclass(frozen=True)
class TraceRefs:
    """Куда трейс привязан: агент новостей, его воркфлоу и ревизии ролей.

    Ревизии ролей приходят из настроек (`.env`). Пустая
    ссылка означает «объект не заведён» — тогда спан всё равно отправляется, просто без этой ссылки:
    трассировка не должна зависеть от того, дошли ли руки до создания объекта.
    """

    workflow_id: str = ""
    workflow_slug: str = ""
    agent_id: str = ""
    #: `роль → id ревизии этой роли`.
    role_revisions: dict[str, str] = field(default_factory=dict)


def trace_refs(settings) -> TraceRefs:
    """Собрать ссылки из настроек. Единственное место, где читаются `OTEL_*` про роли."""
    # Агент выпуска — news-hunter (`OTEL_NEWS_WORKFLOW_*`); если он не задан, откатываемся на общие
    # `OTEL_WORKFLOW_*` — лучше ссылка на агента диалога, чем отсутствие ссылки вовсе (она бы
    # означала «трейс ничей» и не нашлась бы в интерфейсе).
    news_id = getattr(settings, "otel_news_workflow_id", "") or ""
    news_slug = getattr(settings, "otel_news_workflow_slug", "") or ""
    return TraceRefs(
        workflow_id=news_id or getattr(settings, "otel_workflow_id", "") or "",
        workflow_slug=news_slug or getattr(settings, "otel_workflow_slug", "") or "",
        agent_id=getattr(settings, "otel_agent_id", "") or "",
        role_revisions={
            "researcher": getattr(settings, "otel_role_researcher_revision", "") or "",
            "author": getattr(settings, "otel_role_author_revision", "") or "",
            "verifier": getattr(settings, "otel_role_verifier_revision", "") or "",
        },
    )


def release_attributes(*, refs: TraceRefs, chat_id: str = "") -> dict:
    """Нейтральное описание корневого спана выпуска: корень + ссылки на воркфлоу.

    **`agent_id` берётся из настроек, а если его нет — из самого воркфлоу выпуска**: агентом выпуска
    считается `news-hunter`, и пустая ссылка на агента была бы хуже отсутствия — приём складывает
    нераспознанные ключи в `unsupported`, и там появлялся бы мусорный пустой `agent`.

    ❗ Фолбэк живёт ЗДЕСЬ, у вызывающего, а не в адаптере: «агент выпуска — сам воркфлоу» — это наше
    решение о продукте, а не свойство вендора. Потерять его значило бы молча лишить корень привязки, и
    это ровно тот случай, ради которого образец с пустым `agent_id` закреплён тестом.
    """
    return vendors.active().attributes(vendors.SpanShape(
        kind=vendors.WORKFLOW,
        session_id=session_id_for(chat_id),
        agent_id=refs.agent_id or refs.workflow_id,
        workflow_id=refs.workflow_id,
        workflow_slug=refs.workflow_slug,
    ))


def role_attributes(*, refs: TraceRefs, role_key: str) -> dict:
    """Нейтральное описание спана роли: тип «агент» и ссылка на **ревизию** этой роли.

    Ревизия, а не только имя: ревизия — это версия инструкции роли, и именно она делает «своя ревизия
    у каждой роли» проверяемым фактом, а не намерением. Пустая ревизия не мешает: адаптер просто не
    поставит ссылку, и спан всё равно уедет (трассировка не зависит от того, дошла ли рука до
    `agent-roles-setup.py`).
    """
    return vendors.active().attributes(vendors.SpanShape(
        kind=vendors.AGENT,
        revision_id=refs.role_revisions.get(role_key) or "",
    ))
