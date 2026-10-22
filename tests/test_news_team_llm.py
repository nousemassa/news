"""Тесты своего вызова модели для ролей.

**Что здесь проверяется.** Не «работает ли провайдер» (это видно только на настоящем API), а наша
обвязка: форма запроса, цикл инструментов, расход за конкретный вызов, отказы. Модель заменена
двойником транспорта `httpx.MockTransport`, поэтому тесты бесплатны и воспроизводимы, а форма
запроса проверяется ровно та же, что уходит в сеть.
"""
from __future__ import annotations

import json

import httpx
import pytest

from news_agent.news_team import llm
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.roles import AUTHOR, RESEARCHER, VERIFIER, role
from news_agent.news_team.toolset import Tool, local_tools, tools_for_role


class Settings:
    """Двойник настроек: только то, что читает вызов модели.

    `reasoning` здесь — попытка включить рассуждения, которую клиент обязан игнорировать
    (`crew_reasoning_enabled` в настоящих настройках больше нет — см. `config.py`).
    """

    def __init__(self, *, api_key: str = "key", reasoning: bool = False,
                 price_in: float = 0.0, price_out: float = 0.0) -> None:
        self._api_key = api_key
        self.crew_reasoning_enabled = reasoning
        self.team_price_in_per_1k_usd = price_in
        self.team_price_out_per_1k_usd = price_out

    def chat_credentials(self) -> tuple[str, str, str]:
        return "http://provider.test/v1", self._api_key, "model-x"


def answer(*, text: str = "текст заметки", tool_calls: list[dict] | None = None,
           prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def transport(*answers: dict, seen: list[dict] | None = None) -> httpx.MockTransport:
    """Отвечает по очереди; тела запросов складывает в `seen` для проверки формы."""
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=queue.pop(0) if queue else answer())

    return httpx.MockTransport(handler)


def client_with(*answers: dict, seen: list[dict] | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport(*answers, seen=seen))


async def fake_call(args: dict) -> str:
    fake_call.seen = args
    return "результат инструмента"


@pytest.mark.asyncio
async def test_request_shape_carries_reasoning_flag_and_temperature():
    """Рассуждения выключены: с ними модель отдаёт пустой `content`, и шаг падает."""
    seen: list[dict] = []
    async with client_with(answer(text="готово"), seen=seen) as client:
        reply = await llm.ask_role(settings=Settings(reasoning=False), prompt="задание", client=client)

    body = seen[0]
    assert body["model"] == "model-x" and body["temperature"] == llm.TEMPERATURE
    assert body["reasoning"] == {"enabled": False}
    assert body["messages"] == [{"role": "user", "content": "задание"}]
    assert "tools" not in body, "без инструментов поле не нужно"
    assert reply.text == "готово"


@pytest.mark.asyncio
async def test_reasoning_stays_off_even_if_the_settings_say_otherwise():
    """Рассуждения выключены ЖЁСТКО: настройка на них влиять не должна.

    Это сторож от возврата дефекта: с включёнными рассуждениями рассуждающая модель отдаёт весь
    бюджет ответа в `reasoning_tokens`, `content` приходит пустым, и шаг роли падает.
    """
    seen: list[dict] = []
    async with client_with(answer(), seen=seen) as client:
        await llm.ask_role(settings=Settings(reasoning=True), prompt="задание", client=client)
    assert seen[0]["reasoning"] == {"enabled": False}, \
        "даже при True в настройке в теле запроса должно быть false"


@pytest.mark.asyncio
async def test_tools_are_declared_only_when_the_role_has_them():
    seen: list[dict] = []
    tools = {"lost_facts": Tool(name="lost_facts", description="проверка", parameters={"type": "object"},
                                call=fake_call)}
    async with client_with(answer(), seen=seen) as client:
        await llm.ask_role(settings=Settings(), prompt="задание", tools=tools, client=client)
    declared = seen[0]["tools"]
    assert declared[0]["function"]["name"] == "lost_facts"
    assert seen[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_usage_is_per_call_not_accumulated():
    """Расход берётся из ответа, а не копится на объекте модели."""
    async with client_with(answer(prompt_tokens=100, completion_tokens=20),
                           answer(prompt_tokens=7, completion_tokens=3)) as client:
        first = await llm.ask_role(settings=Settings(), prompt="a", client=client)
        second = await llm.ask_role(settings=Settings(), prompt="b", client=client)

    assert first.usage.prompt_tokens == 100 and first.usage.completion_tokens == 20
    assert second.usage.prompt_tokens == 7 and second.usage.completion_tokens == 3, \
        "второй шаг не должен нести расход первого"


@pytest.mark.asyncio
async def test_price_is_none_without_settings_and_number_with_them():
    async with client_with(answer(prompt_tokens=1000, completion_tokens=1000)) as client:
        without = await llm.ask_role(settings=Settings(), prompt="x", client=client)
    async with client_with(answer(prompt_tokens=1000, completion_tokens=1000)) as client:
        with_prices = await llm.ask_role(
            settings=Settings(price_in=0.001, price_out=0.002), prompt="x", client=client)

    assert without.cost_usd is None, "ноль выглядел бы как бесплатная команда"
    assert with_prices.cost_usd == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_tool_call_is_executed_and_its_answer_returned_to_the_model():
    """Цикл инструментов: модель просит инструмент, получает результат и только потом отвечает."""
    seen: list[dict] = []
    request = [{"id": "call-1", "type": "function",
                "function": {"name": "lost_facts", "arguments": json.dumps({"original": "a", "new_text": "b"})}}]
    tools = {"lost_facts": Tool(name="lost_facts", description="", parameters={}, call=fake_call)}
    async with client_with(answer(tool_calls=request, text=""),
                           answer(text="итог", prompt_tokens=1, completion_tokens=1),
                           seen=seen) as client:
        reply = await llm.ask_role(settings=Settings(), prompt="проверь", tools=tools, client=client)

    assert fake_call.seen == {"original": "a", "new_text": "b"}, "аргументы модели дошли до инструмента"
    assert reply.text == "итог" and reply.tool_calls == 1
    second = seen[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call-1"
    assert second[-1]["content"] == "результат инструмента"
    assert reply.usage.total == 17, "расход шага — сумма обоих ответов круга (15 + 2), а не последний ответ"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_as_text_and_does_not_break_the_step():
    request = [{"id": "c1", "type": "function", "function": {"name": "нет_такого", "arguments": "{}"}}]
    seen: list[dict] = []
    async with client_with(answer(tool_calls=request, text=""), answer(text="готово"), seen=seen) as client:
        reply = await llm.ask_role(settings=Settings(), prompt="x", tools={}, client=client)
    assert "нет" in seen[1]["messages"][-1]["content"]
    assert reply.text == "готово"


@pytest.mark.asyncio
async def test_broken_arguments_json_do_not_break_the_step():
    request = [{"id": "c1", "type": "function", "function": {"name": "lost_facts", "arguments": "{не json"}}]
    tools = {"lost_facts": Tool(name="lost_facts", description="", parameters={}, call=fake_call)}
    async with client_with(answer(tool_calls=request, text=""), answer(text="готово")) as client:
        await llm.ask_role(settings=Settings(), prompt="x", tools=tools, client=client)
    assert fake_call.seen == {}, "битая строка аргументов = аргументов нет, а не падение"


@pytest.mark.asyncio
async def test_empty_answer_is_a_failed_step():
    """Пустой `content` без инструментов — отказ шага: публиковать «ничего» нельзя."""
    async with client_with(answer(text="")) as client:
        with pytest.raises(TeamUnavailable, match="пустой ответ"):
            await llm.ask_role(settings=Settings(), prompt="x", client=client)


@pytest.mark.asyncio
async def test_tool_loop_is_bounded():
    request = [{"id": "c1", "type": "function", "function": {"name": "lost_facts", "arguments": "{}"}}]
    tools = {"lost_facts": Tool(name="lost_facts", description="", parameters={}, call=fake_call)}
    answers = [answer(tool_calls=request, text="") for _ in range(llm.MAX_TOOL_ROUNDS + 3)]
    async with client_with(*answers) as client:
        with pytest.raises(TeamUnavailable, match="кругов инструментов"):
            await llm.ask_role(settings=Settings(), prompt="x", tools=tools, client=client)


@pytest.mark.asyncio
async def test_provider_failure_and_timeout_become_team_unavailable():
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет сети", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(TeamUnavailable, match="провайдер недоступен"):
            await llm.ask_role(settings=Settings(), prompt="x", client=client)

    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={}))) as client:
        with pytest.raises(TeamUnavailable, match="503"):
            await llm.ask_role(settings=Settings(), prompt="x", client=client)


@pytest.mark.asyncio
async def test_missing_key_stops_before_any_request():
    async with client_with(answer()) as client:
        with pytest.raises(TeamUnavailable, match="нет ключа"):
            await llm.ask_role(settings=Settings(api_key=""), prompt="x", client=client)


def test_role_gets_only_its_own_tools():
    """Раздача инструментов — часть описания роли, и расхождение проверяется без сети."""
    mcp = {"searxng_web_search": Tool(name="searxng_web_search", description="поиск",
                                      parameters={}, call=fake_call, source="http://mcp")}

    researcher = tools_for_role(definition=role(RESEARCHER), retriever=None, mcp=mcp)
    author = tools_for_role(definition=role(AUTHOR), retriever=None, mcp=mcp)
    verifier = tools_for_role(definition=role(VERIFIER), retriever=None, mcp=mcp)

    assert set(researcher) == {"knowledge_search", "searxng_web_search"}, "исследователю — поиск, в том числе внешний"
    assert author == {}, "у автора инструментов нет"
    assert set(verifier) == {"lost_facts"}, "проверяющему — только проверка фактов"


def test_mcp_tools_are_not_given_to_roles_that_do_not_use_mcp():
    mcp = {"searxng_web_search": Tool(name="searxng_web_search", description="", parameters={}, call=fake_call)}
    verifier = tools_for_role(definition=role(VERIFIER), retriever=None, mcp=mcp)
    assert "searxng_web_search" not in verifier


@pytest.mark.asyncio
async def test_local_tools_speak_to_the_model_in_text():
    tools = local_tools(retriever=None)
    assert await tools["knowledge_search"].call({"query": "нет базы"}) == "", \
        "«знаний нет» — пустая строка, а не ошибка"
    assert await tools["lost_facts"].call({"original": "дом 5", "new_text": "дом 5"}) == "потерь нет"
    lost = await tools["lost_facts"].call({"original": "дом 5, цена 10", "new_text": "дом"})
    assert "потеряны факты" in lost
