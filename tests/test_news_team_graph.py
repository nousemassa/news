"""Тесты графа выпуска: порядок ролей, гейт, свой счёт фактов.

**Что здесь проверяется.** Не провайдер и не решатель (у них свои тесты и свой стенд), а
наши решения: порядок ролей графом, пропуск исследователя на готовом материале, что исходник фактов —
черновик автора, что итоговый текст берётся из служебной строки, что число фактов считаем мы, что
цена прогона не выдумывается, и что падение шага называет роль.

Модель заменена двойником-сценарием: сеть не нужна, тесты бесплатны и воспроизводимы.
"""
from __future__ import annotations

import json

import pytest

from agent_core.decisions.client import Decision
from news_agent.news import extract_facts
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.graph import Answer, TeamDeps, build_news_graph, material_as_draft
from news_agent.news_team.run import run_from_state, run_once, search_news

VERDICT = "VERDICT|"


class Script:
    """Сценарий: что роль ответит и в каком порядке её позвали."""

    def __init__(self, *, answers: dict[str, str], costs: dict[str, float | None] | None = None,
                 tokens: dict[str, int] | None = None, prompts: dict[str, str] | None = None,
                 fail: dict[str, str] | None = None) -> None:
        self.answers = answers
        self.costs = costs or {}
        self.tokens = tokens or {}
        self.prompts = prompts if prompts is not None else {}
        self.fail = fail or {}
        self.order: list[str] = []

    async def __call__(self, role_key: str, prompt: str) -> Answer:
        self.order.append(role_key)
        self.prompts[role_key] = prompt
        if role_key in self.fail:
            raise ValueError(self.fail[role_key])
        text = self.answers.get(role_key, "")
        return Answer(text=text, cost_usd=self.costs.get(role_key, 0.01),
                      tokens=self.tokens.get(role_key, 7))


def news_marker() -> str:
    """Ответ исследователя «ничего не нашёл» — тот же маркер, что у одиночного пути."""
    from news_agent.news import NEWS_NONE

    return NEWS_NONE


def good_news() -> str:
    """Найденная новость в контракте `NEWS|{json}` — иначе выпуск обязан отказать."""
    body = "Цена выросла на двенадцать процентов за квартал, спрос держится на прежнем уровне"
    return "NEWS|" + json.dumps({"title": "Заголовок", "text": body, "url": "https://a.invalid/1",
                                 "source": "источник"}, ensure_ascii=False)


def verdict(text: str, *, lost: list[str] | None = None, show: bool = True) -> str:
    return VERDICT + json.dumps({"text": text, "lost": lost or [], "show": show})


async def run(script: Script, *, material: str = "", gate=None, topic: str = "тема") -> dict:
    graph = build_news_graph(TeamDeps(ask=script, gate_ask=gate))
    return await graph.ainvoke({"topic": topic, "max_items": 1, "material": material,
                                "hint": "", "url": "", "steps": []})


@pytest.mark.asyncio
async def test_roles_run_in_specification_order():
    """Порядок «исследователь → автор → проверяющий» — часть спецификации, и он в рёбрах графа."""
    script = Script(answers={"researcher": good_news(), "author": "черновик",
                             "verifier": verdict("итог")})
    await run(script)
    assert script.order == ["researcher", "author", "verifier"]


@pytest.mark.asyncio
async def test_researcher_is_skipped_when_material_is_given():
    """Готовый материал — исследователя не зовём: иначе сравнение с одиночным агентом нечестное."""
    script = Script(answers={"author": "черновик", "verifier": verdict("итог")})
    await run(script, material="материал из news_drafts")
    assert script.order == ["author", "verifier"]


@pytest.mark.asyncio
async def test_researcher_without_material_stops_the_run():
    script = Script(answers={"researcher": "   ", "author": "черновик", "verifier": verdict("итог")})
    with pytest.raises(TeamUnavailable, match="исследователь не вернул материал"):
        await run(script)


@pytest.mark.asyncio
async def test_marker_of_nothing_found_stops_the_release_instead_of_making_up_text():
    """Маркер «ничего не нашёл» — не материал: выпуск обязан отказать, а не сочинять прозу о пустоте.

    При недоступном поиске исследователь отдаёт `NEWS_NONE`, автор
    переписывал его и получался текст «в исходном тексте нет содержания — передан только маркер».
    Публиковать такое нельзя, и одиночный путь в этом случае честно говорил «не отдал карточек».
    """
    script = Script(answers={"researcher": news_marker(), "author": "проза про отсутствие материала",
                             "verifier": verdict("проза про отсутствие материала")})
    # Маркер `NEWS_NONE` — состояние «поиск сработал, новостей нет»: причина НАЗВАНА своя, а не общая
    # («ни одной новости» говорила то же самое и про прозу, и про отказ источника — то есть не говорила ничего).
    with pytest.raises(TeamUnavailable, match="новостей не нашлось"):
        await run(script)
    assert script.order == ["researcher"], "до автора дело доходить не должно"


@pytest.mark.asyncio
async def test_search_failure_is_named_differently_than_empty_search():
    """Отказ ПОИСКА и пустой поиск — разные причины, и в отчёте они обязаны читаться по-разному.

    Пока оба состояния сходились в одном маркере, отказ источника (лимит запросов, таймаут,
    недоступный сервер) выглядел как «материала нет», и настоящая причина исчезала. Различие держится
    маркером `SEARCH_FAILED`: поиск отвечал «Too many requests», хотя
    сам SearXNG возвращал `200` и 30 результатов.
    """
    from news_agent.news import SEARCH_FAILED

    script = Script(answers={"researcher": f"{SEARCH_FAILED}: searxng_web_search не сработал (timeout)",
                             "author": "проза", "verifier": verdict("проза")})
    with pytest.raises(TeamUnavailable, match="внешний поиск не сработал"):
        await run(script)
    assert script.order == ["researcher"], "при отказе поиска автора не зовём"

    empty = Script(answers={"researcher": news_marker(), "author": "проза", "verifier": verdict("проза")})
    with pytest.raises(TeamUnavailable, match="новостей не нашлось"):
        await run(empty)


@pytest.mark.asyncio
async def test_search_path_validates_the_role_answer_like_the_release_path():
    """БОЕВОЙ путь поиска обязан проверять ответ роли — тем же механизмом, что граф выпуска.

    Граф поиска — это ОДИН узел исследователя, а
    проверка ответа живёт в узле `prepare` графа ВЫПУСКА. Без неё ответ роли (маркер отказа, «ничего не
    нашёл», проза) доезжает до парсера как есть, и бот отвечает владельцу «агент не отдал карточек» —
    сообщение называет не ту причину, хотя поиск в том же прогоне РАБОТАЛ.
    """
    from news_agent.news import SEARCH_FAILED

    with pytest.raises(TeamUnavailable, match="внешний поиск не сработал"):
        await search_news(topic="т", max_items=1,
                          ask_role=Script(answers={"researcher": f"{SEARCH_FAILED}: поиск не сработал"}))
    with pytest.raises(TeamUnavailable, match="новостей не нашлось"):
        await search_news(topic="т", max_items=1, ask_role=Script(answers={"researcher": news_marker()}))
    with pytest.raises(TeamUnavailable, match="ни одной новости"):
        await search_news(topic="т", max_items=1, ask_role=Script(answers={"researcher": "проза без строк"}))
    # А пригодный ответ обязан ПРОЙТИ и доехать строками `NEWS|` — иначе путь поиска сломался бы всегда.
    found = await search_news(topic="т", max_items=1,
                              ask_role=Script(answers={"researcher": good_news()}))
    assert "NEWS|" in found.text


@pytest.mark.asyncio
async def test_empty_author_answer_stops_the_run():
    """Пустой текст автора — отказ шага: публиковать «ничего» нельзя, уходим на одиночный путь."""
    script = Script(answers={"researcher": good_news(), "author": "", "verifier": verdict("итог")})
    with pytest.raises(TeamUnavailable, match="автор вернул пустой текст"):
        await run(script)


@pytest.mark.asyncio
async def test_verdict_text_is_taken_and_report_is_not():
    """В итог идёт ТЕКСТ из служебной строки, а не отчёт роли о проверке."""
    script = Script(answers={"researcher": good_news(), "author": "черновик",
                             "verifier": "Отчёт о проверке: инструмент вернул пустой список.\n"
                                         + verdict("чистый текст заметки")})
    state = await run(script)
    assert state["text"] == "чистый текст заметки"
    assert "Отчёт о проверке" not in state["text"]


@pytest.mark.asyncio
async def test_missing_verdict_marker_stops_the_run():
    script = Script(answers={"researcher": good_news(), "author": "черновик",
                             "verifier": "просто текст без служебной строки"})
    with pytest.raises(TeamUnavailable, match="не дал вердикт"):
        await run(script)


@pytest.mark.asyncio
async def test_facts_are_counted_from_the_source_and_losing_everything_is_not_perfect():
    """Знаменатель — факты ИСХОДНИКА: текст, потерявший всё, не должен давать долю 1.0."""
    source = "Цена выросла на 12 процентов, дом 5"
    script = Script(answers={"author": "новость без цифр", "verifier": verdict("новость без цифр")})
    state = await run(script, material=source)
    run_result = run_from_state(state, topic="тема")

    assert run_result.total_facts == len(extract_facts(source)) > 0
    assert run_result.lost_facts, "потери обязаны быть найдены нашим счётом"
    assert run_result.facts_kept_share < 1.0


@pytest.mark.asyncio
async def test_failure_of_a_step_names_the_role():
    script = Script(answers={"researcher": good_news(), "author": "черновик", "verifier": "x"},
                    fail={"author": "нет ключа"})
    with pytest.raises(TeamUnavailable, match="шаг author упал"):
        await run(script)


@pytest.mark.asyncio
async def test_price_is_unknown_if_any_step_has_no_price():
    """Частичная сумма выглядела бы как полная и занизила бы цену команды."""
    part = Script(answers={"researcher": good_news(), "author": "черновик", "verifier": verdict("итог")},
                  costs={"researcher": 0.01, "author": None, "verifier": 0.02})
    full = Script(answers={"researcher": good_news(), "author": "черновик", "verifier": verdict("итог")},
                  costs={"researcher": 0.01, "author": 0.03, "verifier": 0.02},
                  tokens={"researcher": 7, "author": 11, "verifier": 13})

    partial = run_from_state(await run(part), topic="тема")
    complete = run_from_state(await run(full), topic="тема")

    assert partial.cost_usd is None
    assert complete.cost_usd == pytest.approx(0.06)
    assert complete.tokens == 31, "токены складываются по шагам"
    assert [step.role for step in complete.steps] == ["researcher", "author", "verifier"]


@pytest.mark.asyncio
async def test_seconds_are_measured_by_the_injected_clock():
    ticks = iter([0.0, 1.5, 1.5, 3.0, 3.0, 6.0, 6.0])
    script = Script(answers={"researcher": good_news(), "author": "черновик", "verifier": verdict("итог")})
    graph = build_news_graph(TeamDeps(ask=script, clock=lambda: next(ticks)))
    state = await graph.ainvoke({"topic": "т", "max_items": 1, "material": "", "steps": []})
    assert run_from_state(state, topic="т").seconds == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_material_as_draft_removes_the_envelope_and_keeps_the_address():
    """Обёртка `NEWS|{json}` не идёт в текст: иначе цифры адреса считаются фактами."""
    body = "Цена выросла на двенадцать процентов за квартал, спрос держится на прежнем уровне"
    material = 'NEWS|' + json.dumps({"title": "Заголовок", "text": body, "url": "https://a/789016"})
    text, url = material_as_draft(material)
    assert text == f"Заголовок. {body}" and url == "https://a/789016"
    assert "NEWS|" not in text and "789016" not in text

    script = Script(answers={"author": "черновик", "verifier": verdict("итог")})
    state = await run(script, material=material)
    assert state["draft_url"] == "https://a/789016"
    assert "NEWS|" not in state["source_text"]
    prompt = script.prompts["author"]
    assert "Цена выросла" in prompt, "текст новости доехал до автора"
    assert '"url"' not in prompt and '"title"' not in prompt, \
        "обёртка `NEWS|{json}` до автора не доезжает (слово `NEWS|` в инструкции — это другое)"


class Solver:
    """Двойник локального решателя гейта: отвечает заданным вариантом и уверенностью."""

    def __init__(self, answer: str, confidence: float = 0.95) -> None:
        self.answer = answer
        self.confidence = confidence

    async def __call__(self, text: str, question: str, options: dict[str, str]) -> Decision:
        return Decision(answer=self.answer, top_probability=self.confidence,
                        probabilities={self.answer: self.confidence})


@pytest.mark.asyncio
async def test_confident_local_no_turns_the_show_off_even_if_the_role_says_show():
    """Уверенное локальное «нет» сильнее слова роли — иначе гейт был бы украшением."""
    script = Script(answers={"researcher": good_news(), "author": "черновик",
                             "verifier": verdict("итог", show=True)})
    state = await run(script, gate=Solver("нет", 0.99))
    result = run_from_state(state, topic="т")
    assert result.gate_level > 0 and result.gate_answer == "нет"
    assert result.shown is False


@pytest.mark.asyncio
async def test_gate_failure_does_not_break_the_release():
    async def broken(text: str, question: str, options: dict[str, str]) -> Decision:
        raise RuntimeError("решатель сломался по-настоящему")

    script = Script(answers={"researcher": good_news(), "author": "черновик", "verifier": verdict("итог")})
    state = await run(script, gate=broken)
    assert state["text"] == "итог", "выпуск собирается даже при сбое гейта"
    assert state.get("gate_level", 0) == 0


@pytest.mark.asyncio
async def test_without_a_gate_the_role_decides_about_showing():
    script = Script(answers={"researcher": good_news(), "author": "черновик",
                             "verifier": verdict("итог", show=False)})
    assert run_from_state(await run(script), topic="т").shown is False


@pytest.mark.asyncio
async def test_run_once_needs_either_a_seam_or_settings():
    with pytest.raises(TeamUnavailable, match="ни шва модели, ни настроек"):
        await run_once(topic="т", max_items=1)
