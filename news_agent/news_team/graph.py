"""Новостной контур как граф: три роли, гейт и сборка итога.

**Почему порядок шагов живёт в графе, а не в цикле.** Порядок «исследователь → автор →
проверяющий факты» — часть спецификации пункта, и его надо уметь проверить тестом. Граф делает
порядок явным: узлы и рёбра — это и есть порядок, а не последовательность вызовов внутри функции.
Готовый фреймворк ролей (CrewAI) даёт исполнение роли, но не даёт видеть конвейер: поэтому роли
становятся узлами.

**Идиома — та же, что в графе напоминаний:** состояние — `TypedDict`,
всё внешнее — в `Deps` (ни одного импорта транспорта, иначе не проверить), сборка — функцией, а
`langgraph` импортируется внутри сборки.

**Чекпойнтер здесь НЕ подключается, и это отличие осознанное.** Выпуск живёт секунды:
состояние не должно переживать перезапуск процесса, а неудачный выпуск уходит на одиночный путь.
Граф напоминаний живёт днями и прерывается на человека — там чекпойнты обязательны.

**Что делает наша сторона, а что — роль.** Роль пишет текст. Число фактов, проверку потерь, решение
о показе и учёт расхода считает наш код: словами модели нельзя управлять числом «доля сохранённых
фактов», а решение о публикации не должно зависеть от того, как роль перескажет инструмент.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from agent_core import vendors
from news_agent.news import NEWS_NONE, NEWS_PREFIX, SEARCH_FAILED, extract_facts, parse_news_blocks
from news_agent.news_team import tracing
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.prompts import VERIFIER_TURN, author_turn, researcher_turn
from news_agent.news_team.roles import AUTHOR, RESEARCHER, VERIFIER
from news_agent.news_team.tools import news_gate
from news_agent.news_team.tools.facts import missing_facts
from news_agent.news_team.verdict import VerdictError, parse_verdict

log = logging.getLogger(__name__)

#: Сколько новостей из находок исследователя попадает в ОДИН черновик. Одна заметка на выпуск —
#: контракт существующего пути (`build_rewrite_turn`), и расширять его здесь нельзя: публикация
#: остаётся у новостного контура, а не у команды.
MAX_DRAFT_ITEMS = 1

#: Имена узлов. Роли совпадают с ключами ролей (`roles.py`) — по ним же идут спаны.
NODE_PREPARE = "prepare"
NODE_GATE = "gate"
NODE_ASSEMBLE = "assemble"


class NewsState(TypedDict, total=False):
    """Состояние одного выпуска: от задания до итогового текста и решения о показе."""

    topic: str
    max_items: int
    material: str          # готовый материал, если он есть (тогда исследователь пропускается)
    #: Материал добыла модель (исследователь), а не пришёл готовым. Различие решает, проверять ли
    #: формат: от модели мы требуем строки `NEWS|`, а готовый черновик из `news_drafts` — просто текст.
    found_by_model: bool
    hint: str
    url: str
    source_text: str       # то, что получил автор: черновик без обёртки `NEWS|{json}`
    draft_url: str         # адрес источника отдельно — он не идёт в текст и не считается фактом
    draft: str             # текст автора
    gate_level: int
    gate_answer: str
    gate_confidence: float
    gate_reason: str
    text: str              # итоговый текст от проверяющего
    claimed_lost: tuple[str, ...]   # что роль НАЗВАЛА потерянным (сверяется с нашим счётом)
    claimed_show: bool
    lost: tuple[str, ...]           # что посчитали МЫ
    total_facts: int
    show: bool
    steps: list[dict]               # шаги с расходом и временем — для агрегата `Run`
    tokens: int


@dataclass(frozen=True)
class TeamDeps:
    """Всё внешнее, что нужно графу. Ни одного импорта провайдера и базы — иначе не проверить."""

    #: Шов модели: `(роль, задание) -> Answer`. Двойник в тестах подставляется сюда.
    ask: Callable[[str, str], Awaitable[Answer]]
    #: Шов локального решателя для гейта новости. `None` — гейта нет.
    gate_ask: Any = None
    gate_threshold: float = 0.75
    #: Часы: вынесены в зависимости, чтобы тест мог проверить время шага без реальных задержек.
    clock: Callable[[], float] = field(default=time.perf_counter)
    #: Трассировка: трейсер и ссылки на объекты платформы. `None` — ШТАТНОЕ состояние: спаны
    #: просто не создаются, а при выключенной трассировке работают те же контексты-пустышки.
    tracer: Any = None
    refs: tracing.TraceRefs = field(default_factory=tracing.TraceRefs)


@dataclass(frozen=True)
class Answer:
    """Ответ роли: текст плюс то, что мы знаем о его цене.

    **Почему `cost_usd` может быть `None`, а не нулём.** Ноль означал бы бесплатную команду в
    таблице замера. Отсутствие цены честно показано как `None`; токены при этом известны всегда.

    **Зачем И итог, И разбивка.** Провайдер отдаёт вход и выход РАЗДЕЛЬНО (`llm.Usage`). Схлопнутые
    в один скаляр, они молчат на витрине метрик: приём ждёт разбивку, а скаляр уходит в
    `unsupported`. Итог оставлен рядом с разбивкой:
    его знают те, кто разбивки не знает (двойники в тестах и прямые вызывающие), и терять известное
    число нельзя.
    """

    text: str
    cost_usd: float | None = None
    tokens: int = 0
    #: Расход входа и выхода раздельно, как его отдал провайдер. Пусто ⇒ разбивки нет, и тогда в спан
    #: уйдёт только итог (а не ноль — ноль стёр бы уже накопленную `cumulative`).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Сколько раз роль попросила инструменты за этот шаг. Нужно для честного отчёта: «роль искала
    #: инструментами, а не по памяти» — это утверждение, и оно должно быть проверяемым числом.
    tool_calls: int = 0


def check_researcher_answer(material: str, *, max_items: int) -> None:
    """Отказ с НАЗВАННОЙ причиной, если ответ роли не годится. ОДИН механизм на ДВА графа.

    **Почему отдельной функцией, а не кодом внутри узла `prepare`.** Валидацию ответа исследователя
    требуют ДВА потребителя: граф выпуска (узел `prepare`) и граф поиска (`build_search_graph`) — он же
    боевой путь новостей. Если проверка живёт только в `prepare`, боевой путь отдаёт парсеру что угодно:
    маркер отказа, «ничего не нашёл» или прозу — и всё это превращается в одно «карточек нет» без
    причины. Механизм, нужный двум потребителям, обязан иметь ОДИН вход (копии расходятся молча).

    **Отказы называются РАЗНЫМИ причинами**: «поиск не сработал» (отказ источника), «поиск
    сработал, новостей нет» и «ответ не тем форматом» — не одно и то же и лечатся по-разному.
    """
    text = (material or "").strip()
    if not text:
        raise TeamUnavailable("материала нет: ни готового, ни от исследователя")
    if SEARCH_FAILED in text:
        raise TeamUnavailable(
            f"внешний поиск не сработал ({text[:120]}) — выпуск не состоится; это отказ ИСТОЧНИКА, "
            f"а не отсутствие новостей")
    if NEWS_NONE in text and NEWS_PREFIX not in text:
        raise TeamUnavailable("поиск сработал, но новостей не нашлось (NEWS_NONE) — выпуск не состоится")
    items, _status = parse_news_blocks(text, max_items=max_items)
    if not items:
        raise TeamUnavailable("исследователь не отдал ни одной новости строками NEWS| — "
                              "выпуск не состоится (нет материала, а не «текст без него»)")


def material_as_draft(material: str) -> tuple[str, str]:
    """Материал исследователя как ЧЕРНОВИК: (текст для работы, адрес как ссылка).

    **Зачем отдельным шагом.** Исследователь отвечает строками `NEWS|{json}`, и в них есть `url`.
    Если отдать автору и проверяющему саму обёртку, проверка потерь посчитает **цифры адреса**
    фактами и найдёт «потери», которых нет: проверяющий разбирает адреса вместо заметки. Факты живут
    в ТЕКСТЕ новости, поэтому в работу идёт текст, а адрес идёт ссылкой.

    Если разбор не дал новостей (материал пришёл готовым), возвращаем его как есть: функция обязана
    быть безобидной на любом входе.
    """
    items, status = parse_news_blocks(material, max_items=MAX_DRAFT_ITEMS)
    if status != "ok" or not items:
        return material.strip(), ""
    parts = []
    for item in items:
        head = f"{item.title}. {item.text}".strip()
        if item.source:
            head = f"{head} (источник: {item.source})"
        parts.append(head)
    return "\n\n".join(parts), (items[0].url or "")


def _nodes(deps: TeamDeps) -> dict:
    """Узлы графа — одна реализация на два сборщика: полный выпуск и один поиск.

    Узлы живут вместе, а не в каждом сборщике по своей копии:исследователь и проверяющий — те же роли с
    теми же заданиями, и вторая копия разошлась бы с первой на первой же правке промпта.
    """
    async def step(role_key: str, prompt: str, state: NewsState) -> str:
        """Один шаг роли: спан агента, время, расход и понятная ошибка с именем роли.

        Спан открывается ЗДЕСЬ, потому что только здесь известно, какая роль исполняется: выше это уже
        «шаг», ниже — «вызов модели». Падение помечается на **своём** ребре, а не на корне.
        """
        started = deps.clock()
        with tracing.span(deps.tracer, role_key,
                          tracing.role_attributes(refs=deps.refs, role_key=role_key)) as current:
            try:
                answer = await deps.ask(role_key, prompt)
            except TeamUnavailable as error:
                tracing.mark_error(current, str(error), error_type=type(error).__name__)
                raise
            except Exception as error:
                tracing.mark_error(current, f"{type(error).__name__}: {error}",
                                   error_type=type(error).__name__)
                raise TeamUnavailable(f"шаг {role_key} упал: {type(error).__name__}: {error}") from error
            # Расход — через адаптер вендора: форма метрики вендорская, а записанная здесь литералами
            # и СКАЛЯРОМ она не попадала в витрину метрик приёма вовсе. Предохранитель «трейсера нет»
            # стоит внутри `set_usage`.
            vendors.active().set_usage(current, vendors.Usage(
                total=answer.tokens,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                tool_calls=answer.tool_calls,
            ))
        state.setdefault("steps", []).append({
            "role": role_key,
            "text": answer.text,
            "seconds": deps.clock() - started,
            "cost_usd": answer.cost_usd,
            "tokens": answer.tokens,
            "tool_calls": answer.tool_calls,
        })
        return answer.text

    async def researcher(state: NewsState) -> NewsState:
        found = await step(
            RESEARCHER, researcher_turn(topic=state["topic"], max_items=int(state.get("max_items") or 1)),
            state)
        if not found.strip():
            raise TeamUnavailable("исследователь не вернул материал")
        return {"material": found, "found_by_model": True}

    async def prepare(state: NewsState) -> NewsState:
        """Привести материал к черновику — ВСЕГДА, откуда бы он ни пришёл.

        Нормализация живёт отдельным узлом, а не внутри исследователя: материал может прийти готовым
        (`news_drafts`), и тогда исследователя не зовут вовсе — а работа с обёрткой `NEWS|{json}`
        нужна ровно так же. Забыть её там, где исследователя пропустили, значит отдать автору пустой
        текст и посчитать фактами цифры адресов.
        """
        with tracing.span(deps.tracer, NODE_PREPARE, tracing.node_attributes(name=NODE_PREPARE)):
            material = str(state.get("material") or "").strip()
            if not material:
                raise TeamUnavailable("материала нет: ни готового, ни от исследователя")
            if state.get("found_by_model"):
                # **Проверка на недоступный поиск.** Когда поиск недоступен, исследователь честно
                # отвечает маркером «ничего не нашёл» — но без этой проверки выпуск продолжался: автор
                # переписывал маркер и выдавал правдоподобную прозу О ТОМ, что материала нет («в исходном
                # тексте нет содержания — передан только маркер»). Публикация такого текста — худший исход,
                # чем отказ: в одиночном пути этот случай честно давал «не отдал карточек».
                #
                # **Два отказа называются РАЗНЫМИ причинами**. «Поиск не сработал» (лимит,
                # таймаут, недоступный сервер) и «поиск сработал, новостей нет» — разные состояния, и в
                # отчёте они обязаны читаться по-разному: пока оба давали одну строку про материал,
                # причина отказа исчезала, а диагностика называла не то, что случилось.
                # Проверка ответа роли — ОБЩАЯ с графом поиска (боевым путём новостей): см.
                # `check_researcher_answer`. Пока она жила только здесь, боевой путь отдавал парсеру
                # что угодно (маркер отказа, «ничего не нашёл», прозу) — и без причины.
                check_researcher_answer(material, max_items=MAX_DRAFT_ITEMS)
            source_text, draft_url = material_as_draft(material)
            return {"source_text": source_text, "draft_url": draft_url}

    async def author(state: NewsState) -> NewsState:
        draft = await step(
            AUTHOR,
            author_turn(draft_text=state.get("source_text") or "",
                        hint=str(state.get("hint") or ""),
                        url=str(state.get("url") or state.get("draft_url") or "")),
            state)
        # Пустой ответ автора — отказ шага, а не пустая публикация: у рассуждающей модели это самый
        # частый способ вернуть пустоту, и вызывающий обязан уйти на одиночный путь.
        if not draft.strip():
            raise TeamUnavailable("автор вернул пустой текст")
        return {"draft": draft}

    async def gate(state: NewsState) -> NewsState:
        """Гейт новости спрашивает НАШ код, а не роль: решение о показе должно быть
        проверяемым. Сбой гейта прогон не роняет — шаг идёт без гейта, причина в журнал."""
        with tracing.span(deps.tracer, NODE_GATE, tracing.node_attributes(name=NODE_GATE)):
            if deps.gate_ask is None:
                return {}
            try:
                decided = await news_gate.ask_gate(state["draft"], ask=deps.gate_ask,
                                                   threshold=deps.gate_threshold)
            except Exception as error:  # noqa: BLE001 — гейт не должен уносить выпуск
                log.error("команда: гейт новости упал (%s: %s) — выпуск идёт без гейта",
                          type(error).__name__, error)
                return {}
            return {"gate_level": decided.level, "gate_answer": decided.answer,
                    "gate_confidence": decided.confidence, "gate_reason": decided.reason}

    async def verifier(state: NewsState) -> NewsState:
        gate_text = ""
        if state.get("gate_answer"):
            decided = news_gate.Step(level=int(state.get("gate_level") or 0),
                                     answer=str(state.get("gate_answer") or ""),
                                     confidence=float(state.get("gate_confidence") or 0.0),
                                     reason=str(state.get("gate_reason") or ""))
            gate_text = f"\n\n## Гейт новости (уровень лестницы {decided.level})\n" \
                        f"{news_gate.format_for_model(decided)}"
        answer = await step(
            VERIFIER,
            # Исходник — ЧЕРНОВИК, а не обёртка `NEWS|{json}`: сравнивать надо то, что получил автор.
            f"{VERIFIER_TURN}{gate_text}"
            f"\n\n## Исходник\n{state.get('source_text') or ''}"
            f"\n\n## Проверяемый текст\n{state.get('draft') or ''}",
            state)
        try:
            text, claimed_lost, claimed_show = parse_verdict(answer)
        except VerdictError as error:
            raise TeamUnavailable(f"проверяющий не дал вердикт: {error}") from error
        return {"text": text, "claimed_lost": claimed_lost, "claimed_show": claimed_show}

    async def assemble(state: NewsState) -> NewsState:
        """Итог считает наш код: факты — из исходника, потери — тем же критерием, что у инструмента.

        Слово роли сверяется с нашим счётом: расхождение видно в журнале, и разбирается человеком,
        а не молчанием. Показ новости решает гейт: если он прошёл, слово роли учитывается, если нет —
        его вердикт сильнее."""
        with tracing.span(deps.tracer, NODE_ASSEMBLE, tracing.node_attributes(name=NODE_ASSEMBLE)):
            source = str(state.get("source_text") or "")
            lost = tuple(missing_facts(source, str(state.get("text") or "")))
            if bool(state.get("claimed_lost")) != bool(lost):
                log.warning("команда: вердикт роли расходится со счётом (роль: %s, счёт: %s)",
                            len(state.get("claimed_lost") or ()), len(lost))
            steps = state.get("steps") or []
            shown = bool(state.get("claimed_show", True))
            if state.get("gate_answer"):
                decided = news_gate.Step(level=int(state.get("gate_level") or 0),
                                         answer=str(state.get("gate_answer") or ""),
                                         confidence=float(state.get("gate_confidence") or 0.0),
                                         reason=str(state.get("gate_reason") or ""))
                shown = news_gate.apply(decided, shown)
            return {"lost": lost, "total_facts": len(extract_facts(source)), "show": shown,
                    "tokens": sum(int(item.get("tokens") or 0) for item in steps)}

    return {
        "researcher": researcher,
        "prepare": prepare,
        "author": author,
        "gate": gate,
        "verifier": verifier,
        "assemble": assemble,
    }


def build_news_graph(deps: TeamDeps):
    """Собрать граф выпуска: `START → researcher? → prepare → author → gate → verifier → assemble → END`.

    Исследователь пропускается, когда материал уже есть. Это не «гибкость», а условие честного
    сравнения: сравнивать команду с одиночным агентом можно только на ОДНИХ И ТЕХ ЖЕ входах.
    """
    from langgraph.graph import END, START, StateGraph

    nodes = _nodes(deps)
    def after_start(state: NewsState) -> str:
        return NODE_PREPARE if str(state.get("material") or "").strip() else RESEARCHER

    graph = StateGraph(NewsState)
    graph.add_node(RESEARCHER, nodes["researcher"])
    graph.add_node(NODE_PREPARE, nodes["prepare"])
    graph.add_node(AUTHOR, nodes["author"])
    graph.add_node(NODE_GATE, nodes["gate"])
    graph.add_node(VERIFIER, nodes["verifier"])
    graph.add_node(NODE_ASSEMBLE, nodes["assemble"])
    graph.add_conditional_edges(START, after_start,
                                {RESEARCHER: RESEARCHER, NODE_PREPARE: NODE_PREPARE})
    graph.add_edge(RESEARCHER, NODE_PREPARE)
    graph.add_edge(NODE_PREPARE, AUTHOR)
    graph.add_edge(AUTHOR, NODE_GATE)
    graph.add_edge(NODE_GATE, VERIFIER)
    graph.add_edge(VERIFIER, NODE_ASSEMBLE)
    graph.add_edge(NODE_ASSEMBLE, END)
    return graph.compile()


def build_search_graph(deps: TeamDeps):
    """Собрать граф ОДНОГО поиска: `START → researcher → END`.

    Нужен боевому новостному пути: он берёт из прогона только строки `NEWS|{json}` и сам решает,
    что делать с карточками (дедуп, черновики, кнопки). Автор и проверяющий относятся не к поиску,
    а к правке черновика, поэтому здесь их нет.
    """
    from langgraph.graph import END, START, StateGraph

    nodes = _nodes(deps)
    graph = StateGraph(NewsState)
    graph.add_node(RESEARCHER, nodes["researcher"])
    graph.add_edge(START, RESEARCHER)
    graph.add_edge(RESEARCHER, END)
    return graph.compile()
