"""Дерево спанов выпуска: видимость агентов и пайплайна проверяется без сети.

**Что здесь проверяется.** Не «отправились ли спаны в платформу» (это делает живая приёмка), а
**форма дерева**, которую платформа прочитает: один корень-выпуск, три ребёнка-агента с именами ролей,
ссылки на ревизии ролей у детей, тип `task` у узлов, которые ролями не являются, и падение **на своём**
ребре. Плюс главное правило приватности: в трейс уходит ХЕШ чата, а не идентификатор человека.

Форма атрибутов взята не из головы: она измерена пробой приёма — тип спана читается только из
вложенного словаря `ag`, а ссылки работают плоскими ключами `ag.references.*`. Здесь это
зафиксировано тестом, чтобы форма не разъехалась с реальностью.
"""
from __future__ import annotations

import json

import pytest

from agent_core.ids import chat_hash
from news_agent.news_team import tracing
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.graph import Answer
from news_agent.news_team.run import run_once

CHAT = "777"
ROLE_REVISIONS = {
    "researcher": "rev-researcher",
    "author": "rev-author",
    "verifier": "rev-verifier",
}


class Settings:
    """Двойник настроек: нужны только ссылки трассировки."""

    otel_workflow_id = "wf-1"
    otel_workflow_slug = "news-hunter"
    otel_agent_id = "agent-1"
    otel_role_researcher_revision = ROLE_REVISIONS["researcher"]
    otel_role_author_revision = ROLE_REVISIONS["author"]
    otel_role_verifier_revision = ROLE_REVISIONS["verifier"]


def news_line() -> str:
    body = "Цена выросла на двенадцать процентов за квартал, спрос держится на прежнем уровне"
    return "NEWS|" + json.dumps({"title": "Заголовок", "text": body, "url": "https://a.invalid/1",
                                 "source": "источник"}, ensure_ascii=False)


def verdict(text: str) -> str:
    return "VERDICT|" + json.dumps({"text": text, "lost": [], "show": True})


def script(*, fail: str | None = None):
    """Двойник модели: роли отвечают по контракту, одна может упасть."""
    async def ask(role_key: str, prompt: str) -> Answer:
        if role_key == fail:
            raise TeamUnavailable(f"шаг {role_key} упал: проверка сценария")
        if role_key == "researcher":
            return Answer(text=news_line(), cost_usd=0.01, tokens=11, tool_calls=4)
        if role_key == "author":
            return Answer(text="новость без цифр", cost_usd=0.01, tokens=22)
        return Answer(text=verdict("новость без цифр"), cost_usd=0.01, tokens=33, tool_calls=1)

    return ask


@pytest.fixture
def exporter_and_tracer():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider.get_tracer("test")


def by_name(spans) -> dict:
    return {span.name: span for span in spans}


@pytest.mark.asyncio
async def test_tree_has_one_root_and_three_role_children(exporter_and_tracer):
    exporter, tracer = exporter_and_tracer
    await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(),
                   chat_id=CHAT, tracer=tracer)

    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 1, "корень трейса ровно один — иначе иерархия в интерфейсе развалится"
    root = roots[0]
    assert root.name == tracing.RELEASE_SPAN

    names = by_name(spans)
    for role in ("researcher", "author", "verifier"):
        child = names[role]
        assert child.parent is not None and child.parent.span_id == root.context.span_id, \
            f"роль {role} обязана быть ребёнком выпуска"


@pytest.mark.asyncio
async def test_role_spans_carry_agent_type_and_their_own_revision(exporter_and_tracer):
    exporter, tracer = exporter_and_tracer
    await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(),
                   chat_id=CHAT, tracer=tracer)
    names = by_name(exporter.get_finished_spans())

    for role, revision in ROLE_REVISIONS.items():
        attributes = names[role].attributes
        assert attributes["ag"]["type"]["span"] == "agent", "тип агента — из вложенного словаря `ag`"
        assert attributes["ag.references.workflow_revision.id"] == revision, \
            "у роли ссылка на СВОЮ ревизию — иначе «своя ревизия у роли» только слова"


@pytest.mark.asyncio
async def test_root_carries_workflow_reference_and_session_hash(exporter_and_tracer):
    exporter, tracer = exporter_and_tracer
    await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(),
                   chat_id=CHAT, tracer=tracer)
    root = by_name(exporter.get_finished_spans())[tracing.RELEASE_SPAN]

    assert root.attributes["ag"]["type"]["span"] == "workflow"
    assert root.attributes["ag.references.workflow.id"] == "wf-1"
    assert root.attributes["ag.references.workflow.slug"] == "news-hunter"
    assert root.attributes["ag"]["session"]["id"] == chat_hash(CHAT)
    assert CHAT not in json.dumps(root.attributes, default=str), \
        "в трейс уходит хеш чата, а не идентификатор человека"


@pytest.mark.asyncio
async def test_nodes_that_are_not_roles_are_tasks_without_agent(exporter_and_tracer):
    """`prepare`, `gate`, `assemble` — не агенты: иначе «три агента» читались бы как шесть."""
    exporter, tracer = exporter_and_tracer
    await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(),
                   chat_id=CHAT, tracer=tracer)
    names = by_name(exporter.get_finished_spans())

    for node in tracing.NON_ROLE_NODES:
        assert names[node].attributes["ag"]["type"]["span"] == "task", f"{node} — не агент"
        assert "ag.references.workflow_revision.id" not in names[node].attributes


@pytest.mark.asyncio
async def test_failed_step_marks_its_own_span_and_the_release_as_failed(exporter_and_tracer):
    """Падение шага видно на СВОЁМ ребре, а корень помечен как неудавшийся выпуск.

    **Почему красный и корень.** Это не «ошибка размазалась»: выпуск действительно не состоялся —
    он не отдал заметку. Разделение ролей при этом сохраняется: у ребра в описании статуса стоит
    имя упавшей роли, поэтому понять «где упало» можно по дереву, а не по тексту журнала.
    """
    exporter, tracer = exporter_and_tracer
    with pytest.raises(TeamUnavailable):
        await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(fail="author"),
                       chat_id=CHAT, tracer=tracer)

    names = by_name(exporter.get_finished_spans())
    from opentelemetry.trace import StatusCode

    assert names["author"].status.status_code == StatusCode.ERROR
    assert "author" in (names["author"].status.description or ""), "видно, КАКАЯ роль упала"
    # Стандарт OTEL («Recording Errors») требует ДВУХ вещей: статус `Error` И атрибут `error.type`
    # (низкокардинальный признак — каноническое имя исключения). Без второго приёмник видит «упало»,
    # но не видит «чем», и различить отказы можно только по свободному тексту, который для группировки
    # не годится. Проверяем обе стороны: у упавшего признак ЕСТЬ, у соседа его НЕТ.
    assert names["author"].attributes["error.type"] == "TeamUnavailable", \
        "признак ошибки обязан быть, а не только статус"
    assert names["researcher"].status.status_code == StatusCode.UNSET, \
        "упавшая роль не красит соседей"
    assert "error.type" not in names["researcher"].attributes, \
        "у целого соседа признака ошибки быть не должно"
    assert names[tracing.RELEASE_SPAN].status.status_code == StatusCode.ERROR, \
        "выпуск без результата — неудавшийся выпуск"


@pytest.mark.asyncio
async def test_role_span_has_metrics_of_the_step(exporter_and_tracer):
    """Токены и вызовы инструментов попадают в спан: «роль искала инструментами» — проверяемое число.

    **Форма токенов выбрана осознанно, и это не подгонка теста под код.** Скаляр
    (`ag.metrics.tokens = 11`) приём вендора не читает: он ждёт СЛОВАРЬ-разбивку, скаляр уходит в
    `unsupported`, слот остаётся пустым `{}`, и числа ролей в витрину метрик не попадают. Тест
    держит форму, которую приём действительно читает: `incremental`.

    Разбивки у двойника нет (он задаёт только итог), поэтому проверяем ветку «известен только итог» —
    и это отдельный полезный случай: потерять итог из-за отсутствия разбивки было бы хуже, чем было.

    `tool_calls` идёт под НАШИМ префиксом и ставится даже при нуле: слота метрики для него у приёмника
    нет вовсе, и в `unsupported` он попадать не должен — опираться на ведро того, чего приёмник не
    понял, нельзя. Замер: ключ вне `ag` приём ХРАНИТ как обычный атрибут. Ноль здесь осмыслен —
    «инструменты не звались».
    """
    exporter, tracer = exporter_and_tracer
    await run_once(topic="ставка", max_items=1, settings=Settings(), ask_role=script(),
                   chat_id=CHAT, tracer=tracer)
    names = by_name(exporter.get_finished_spans())

    assert names["researcher"].attributes["agenta.tool_calls"] == 4
    assert names["researcher"].attributes["ag.metrics.tokens"] == {"incremental": {"total": 11}}
    assert names["author"].attributes["agenta.tool_calls"] == 0


@pytest.mark.asyncio
async def test_without_tracer_the_release_runs_unchanged(exporter_and_tracer):
    """Выключенная трассировка — штатное состояние: выпуск работает, спанов нет."""
    exporter, _ = exporter_and_tracer
    run = await run_once(topic="ставка", max_items=1, settings=None, ask_role=script())

    assert run.text == "новость без цифр"
    assert exporter.get_finished_spans() == (), "без трейсера спаны не создаются"


class SettingsWithoutAgent(Settings):
    """Двойник без `otel_agent_id`: проверяем фолбэк «агент выпуска — сам воркфлоу»."""

    otel_agent_id = ""


def test_release_agent_falls_back_to_the_workflow():
    """Агент выпуска не задан ⇒ им считается воркфлоу, и привязка НЕ теряется.

    Фолбэк `agent_id or workflow_id` живёт у ВЫЗЫВАЮЩЕГО — потеряй его кто-то при правке, `ag.agent.id`
    исчез бы молча, и корень выпуска остался бы без привязки к агенту (в интерфейсе — «трейс ничей»).
    Пустой `agent_id` — проверяемый случай, а не подразумеваемый.
    """
    refs = tracing.trace_refs(SettingsWithoutAgent())

    attributes = tracing.release_attributes(refs=refs, chat_id=CHAT)

    assert attributes["ag"]["agent"] == {"id": "wf-1"}, "агентом выпуска становится сам воркфлоу"
    assert attributes["ag"]["session"] == {"id": chat_hash(CHAT)}
    assert attributes["ag.references.workflow.id"] == "wf-1"
    assert attributes["ag.references.workflow.slug"] == "news-hunter"
