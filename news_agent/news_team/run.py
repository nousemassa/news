"""Один выпуск команды: сборка графа, вызов и агрегат чисел.

**Почему порядок шагов живёт в графе, а не здесь.** Порядок «исследователь → автор → проверяющий»
задан спецификацией пункта, и он же — то, что мы обязаны уметь проверить тестом. Он записан рёбрами
графа (`graph.py`), а этот модуль делает две другие вещи: собирает зависимости (шов модели,
инструменты, гейт) и **считает числа агрегата** — время, расход и долю сохранённых фактов.

**Что считается нашим кодом, а не берётся из ответа роли.** Число фактов в исходнике и потери —
тем же критерием, что у инструмента проверяющего (`missing_facts`): одним числом «доля сохранённых
фактов» нельзя управлять словами модели, иначе замер перестаёт быть замером.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_core.decisions.ladder import DEFAULT_THRESHOLD
from news_agent.news_team import llm, tracing
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.graph import (
    Answer,
    TeamDeps,
    build_news_graph,
    build_search_graph,
    check_researcher_answer,
)
from news_agent.news_team.roles import role as get_role
from news_agent.news_team.toolset import mcp_toolbox, tools_for_role

log = logging.getLogger(__name__)

#: Роли в порядке спецификации — этот порядок проверяется тестом.
ROLE_ORDER = ("researcher", "author", "verifier")


@dataclass(frozen=True)
class Step:
    """Шаг прогона: кто, сколько занял, сколько стоил и сколько токенов."""

    role: str
    text: str
    seconds: float
    cost_usd: float | None
    tokens: int
    tool_calls: int = 0


@dataclass(frozen=True)
class Run:
    """Итог выпуска. Числа для сравнения считаются здесь, а не в измерительном скрипте."""

    topic: str
    steps: tuple[Step, ...]
    text: str
    seconds: float
    cost_usd: float | None
    tokens: int
    total_facts: int
    lost_facts: tuple[str, ...]
    #: Сколько раз роли звали инструменты за прогон. Ноль здесь — не «мелочь», а сигнал: роль
    #: работала по памяти модели, и это надо видеть в отчёте, а не догадываться.
    tool_calls: int = 0
    # Уровень лестницы и решение гейта: чем закрыт случай и показывать ли выпуск. Хранится
    # в прогоне, потому что «доля случаев без облака» — число, которое требуют документы пункта,
    # и считать его надо по факту, а не по словам роли.
    gate_level: int = 0
    gate_answer: str = ""
    gate_confidence: float = 0.0
    gate_reason: str = ""
    shown: bool = True

    @property
    def facts_kept_share(self) -> float:
        """Доля сохранённых фактов — настоящая доля, а не признак «всё/ничего».

        Знаменатель — число фактов в ИСХОДНИКЕ, а не «сколько нашёл проверяющий»: иначе текст,
        потерявший абсолютно всё, давал бы идеальную долю, потому что проверять в нём нечего.
        Если фактов в исходнике нет вовсе — доля считается сохранённой: терять было нечего.
        """
        if self.total_facts == 0:
            return 1.0
        return (self.total_facts - len(self.lost_facts)) / self.total_facts


def build_ask(*, settings, retriever=None, mcp: dict | None = None):
    """Собрать шов модели: роль получает свои инструменты, вызов идёт через `llm.ask_role`.

    Отдельная фабрика, а не логика внутри узла: раздача инструментов и вызов провайдера — разные
    причины меняться, и в тестах шов подменяется целиком, без сети и без инструментов.
    """
    async def ask(role_key: str, prompt: str):
        tools = tools_for_role(definition=get_role(role_key), retriever=retriever, mcp=mcp)
        reply = await llm.ask_role(settings=settings, prompt=prompt, tools=tools)
        # Разбивку входа/выхода переносим КАК ЕСТЬ: схлопнув её здесь, мы теряем её до спана, и
        # витрина метрик остаётся пустой. Итог оставляем рядом — его знают все.
        return Answer(text=reply.text, cost_usd=reply.cost_usd, tokens=reply.usage.total,
                      prompt_tokens=reply.usage.prompt_tokens,
                      completion_tokens=reply.usage.completion_tokens,
                      tool_calls=reply.tool_calls)

    return ask


async def run_once(
    *,
    topic: str,
    max_items: int,
    settings=None,
    ask_role: Callable[[str, str], Awaitable[Answer]] | None = None,
    material: str = "",
    hint: str = "",
    url: str = "",
    gate_ask=None,
    gate_threshold: float = DEFAULT_THRESHOLD,
    retriever=None,
    mcp: dict | None = None,
    chat_id: str = "",
    tracer=None,
) -> Run:
    """Прогнать выпуск графом: три роли, гейт и наш счёт фактов.

    `ask_role(role_key, prompt) -> Answer` — шов модели; в тестах подставляется двойник, и тогда
    проверяется граф, а не провайдер. Без шва собирается настоящий: инструменты роли + `llm.ask_role`.

    `material` — когда материал уже есть: тогда исследователь пропускается, и выпуск начинается с
    автора. Это нужно не «для гибкости», а для честного сравнения с одиночным агентом на одних и тех
    же входах.

    **Сессия MCP открывается на ВЕСЬ выпуск** (`mcp_toolbox`), а не на каждый вызов инструмента: у
    обёртки поиска лимит считается по инициализациям сессий, и «сессия на вызов» упирается в
    него после нескольких выпусков.
    """
    if ask_role is None and settings is None:
        raise TeamUnavailable("нет ни шва модели, ни настроек — выпуск не собрать")
    if ask_role is not None:
        return await _run_once(topic=topic, max_items=max_items, settings=settings, ask_role=ask_role,
                               material=material, hint=hint, url=url, gate_ask=gate_ask,
                               gate_threshold=gate_threshold, chat_id=chat_id, tracer=tracer)
    async with mcp_toolbox(settings=settings, mcp=mcp) as box:
        return await _run_once(
            topic=topic, max_items=max_items, settings=settings,
            ask_role=build_ask(settings=settings, retriever=retriever, mcp=box.tools()),
            material=material, hint=hint, url=url, gate_ask=gate_ask,
            gate_threshold=gate_threshold, chat_id=chat_id, tracer=tracer)


async def _run_once(
    *,
    topic: str,
    max_items: int,
    settings=None,
    ask_role: Callable[[str, str], Awaitable[Answer]],
    material: str = "",
    hint: str = "",
    url: str = "",
    gate_ask=None,
    gate_threshold: float = DEFAULT_THRESHOLD,
    chat_id: str = "",
    tracer=None,
) -> Run:
    """Сам выпуск: граф, корневой спан и сборка итога. Шов модели обязан быть готовым к этому моменту."""
    tracer = tracer if tracer is not None else _tracer(settings)
    refs = tracing.trace_refs(settings) if settings is not None else tracing.TraceRefs()

    graph = build_news_graph(TeamDeps(ask=ask_role, gate_ask=gate_ask,
                                      gate_threshold=gate_threshold, tracer=tracer, refs=refs))
    initial = {
        "topic": topic,
        "max_items": max_items,
        "material": material,
        "hint": hint,
        "url": url,
        "steps": [],
    }
    # Корневой спан выпуска: дети-роли открываются ВНУТРИ него, поэтому трейс читается как
    # «выпуск → агенты», а не как три отдельные записи.
    with tracing.span(tracer, tracing.RELEASE_SPAN,
                      tracing.release_attributes(refs=refs, chat_id=chat_id)) as current:
        try:
            final = await graph.ainvoke(initial)
        except Exception as error:
            tracing.mark_error(current, f"{type(error).__name__}: {error}",
                               error_type=type(error).__name__)
            raise
    return run_from_state(final, topic=topic)


@dataclass(frozen=True)
class Search:
    """Итог одного поиска: строки `NEWS|{json}` и то, что мы знаем о его цене.

    Отдельный тип, а не строка: боевому пути нужна ещё и стоимость прогона, иначе учёт
    `news_cost_usd` тихо перестал бы работать на новом пути (цена `None` — это «неизвестно», а не ноль).
    """

    text: str
    cost_usd: float | None
    tokens: int
    seconds: float


async def search_news(
    *,
    topic: str,
    max_items: int,
    settings=None,
    ask_role: Callable[[str, str], Awaitable[Answer]] | None = None,
    retriever=None,
    mcp: dict | None = None,
) -> Search:
    """Прогнать ТОЛЬКО поиск: роль исследователя с её инструментами.

    Боевой новостной путь берёт отсюда строки `NEWS|{json}` и дальше решает сам (дедуп, черновики,
    карточки, кнопки) — эти решения принадлежат новостному контуру, а не команде. Автор и
    проверяющий в поиске не участвуют: они относятся к пути правки.

    Сессия MCP — одна на поиск (см. `run_once`): иначе серия выпусков упирается в лимит инициализаций.
    """
    if ask_role is None:
        if settings is None:
            raise TeamUnavailable("нет ни шва модели, ни настроек — поиск не собрать")
        async with mcp_toolbox(settings=settings, mcp=mcp) as box:
            return await _search_news(
                topic=topic, max_items=max_items,
                ask_role=build_ask(settings=settings, retriever=retriever, mcp=box.tools()))
    return await _search_news(topic=topic, max_items=max_items, ask_role=ask_role)


async def _search_news(
    *,
    topic: str,
    max_items: int,
    ask_role: Callable[[str, str], Awaitable[Answer]],
) -> Search:
    """Сам поиск: один узел-роль и сборка строк `NEWS|`."""
    graph = build_search_graph(TeamDeps(ask=ask_role))
    final = await graph.ainvoke({"topic": topic, "max_items": max_items, "material": "", "steps": []})
    text = str(final.get("material") or "")
    # ❗ Ответ роли проверяется ЗДЕСЬ независимо от роли: граф поиска — это один узел исследователя, а
    # проверка формата жила только в узле `prepare` графа ВЫПУСКА. Пока её тут не было, боевой путь
    # отдавал парсеру что угодно (маркер отказа, «ничего не нашёл», прозу) — и всё это становилось
    # «карточек нет» без причины. Механизм проверки — ОДИН (`check_researcher_answer`).
    check_researcher_answer(text, max_items=max_items)
    steps = final.get("steps") or []
    first = steps[0] if steps else {}
    return Search(
        text=str(final.get("material") or ""),
        cost_usd=first.get("cost_usd"),
        tokens=int(first.get("tokens") or 0),
        seconds=float(first.get("seconds") or 0.0),
    )


def _tracer(settings):
    """Трейсер из единственного входа трассировки (`telemetry.setup_telemetry`).

    Выключенная трассировка — это пустышка-`ProxyTracer`, а без настроек — `None`; в обоих случаях
    контексты спанов работают как «ничего не делать», поэтому проверок «включено ли» в коде нет.
    """
    if settings is None:
        return None
    try:
        from agent_core import telemetry

        telemetry.register_roles("news")
        return telemetry.setup_telemetry(settings, agent="news")
    except Exception as error:  # noqa: BLE001 — ошибка трейсера не ломает выпуск
        log.warning("команда: трейсер не собран (%s: %s) — выпуск без трассировки",
                    type(error).__name__, error)
        return None


def run_from_state(state: dict, *, topic: str) -> Run:
    """Собрать агрегат из состояния графа. Отдельно от прогона — чтобы проверялось без модели."""
    steps = []
    for item in state.get("steps") or []:
        steps.append(Step(role=str(item.get("role")), text=str(item.get("text") or ""),
                          seconds=float(item.get("seconds") or 0.0),
                          cost_usd=item.get("cost_usd"), tokens=int(item.get("tokens") or 0),
                          tool_calls=int(item.get("tool_calls") or 0)))
    ordered = tuple(sorted(steps, key=lambda step: ROLE_ORDER.index(step.role)
                           if step.role in ROLE_ORDER else len(ROLE_ORDER)))
    return Run(
        topic=topic,
        steps=ordered,
        text=str(state.get("text") or ""),
        seconds=sum(step.seconds for step in steps),
        # Стоимость известна, только если её назвали ВСЕ шаги: частичная сумма выглядела бы как
        # полная и занизила бы цену команды.
        cost_usd=(sum(step.cost_usd for step in steps)
                  if steps and all(step.cost_usd is not None for step in steps) else None),
        tokens=int(state.get("tokens") or 0),
        total_facts=int(state.get("total_facts") or 0),
        lost_facts=tuple(state.get("lost") or ()),
        tool_calls=sum(step.tool_calls for step in steps),
        gate_level=int(state.get("gate_level") or 0),
        gate_answer=str(state.get("gate_answer") or ""),
        gate_confidence=float(state.get("gate_confidence") or 0.0),
        gate_reason=str(state.get("gate_reason") or ""),
        shown=bool(state.get("show", True)),
    )
