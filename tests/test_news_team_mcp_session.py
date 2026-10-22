"""Сессия MCP и поведение при отказе инструмента — без сети и без настоящего сервера.

**Что здесь проверяется и почему именно это.** Живой прогон показал дефект: выпуск открывал сессию
MCP на КАЖДЫЙ вызов инструмента, а обёртка поиска считает не запросы, а инициализации сессий (20 в
минуту) — несколько выпусков подряд упирались в лимит, и источник отвечал «Too many requests»,
хотя сам поиск работал (`200`, 30 результатов). Отсюда три свойства, каждое со своим тестом:

1. **Одна сессия на выпуск** — инициализаций ровно столько, сколько серверов, а не «плюс по одной на
   каждый вызов». Проверяется счётчиком подставного клиента.
2. **Временный отказ повторяется**, постоянный — нет: неизвестный инструмент не пройдёт и на третий
   раз, а лимит отпускает через секунды.
3. **Отказ инструмента отличим от пустого результата**: наружу идёт маркер `SEARCH_FAILED`, а не
   «инструмент не сработал» без метки и не пустая строка. Иначе роль честно отвечает «ничего не
   нашёл», и отказ источника выглядит как отсутствие материала.

Подставной клиент — это подмена ТРАНСПОРТА, а не проверяемой логики: `McpToolbox` настоящий, и
проверяются именно его решения (когда открыть, когда повторить, что вернуть). Отвечает ли сервер,
подставной клиент не докажет: это проверяется на настоящем сервере.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from news_agent.news import SEARCH_FAILED
from news_agent.news_team import toolset
from news_agent.news_team.tools.mcp import McpServer
from news_agent.news_team.toolset import McpToolbox, _looks_broken, _transient


class _Tool:
    """Объявленный инструмент так, как его отдаёт клиент MCP."""

    def __init__(self, name: str, schema: dict | None = None) -> None:
        self.name = name
        self.description = "поиск в интернете"
        self.inputSchema = schema or {"type": "object", "properties": {"query": {"type": "string"}}}


class _FakeClient:
    """Подставной клиент MCP: считает инициализации и отыгрывает сценарий вызовов.

    Сценарий — очередь: элемент либо исключение (его «поднимает» вызов), либо текст ответа.
    """

    opened: ClassVar[list[str]] = []
    script: ClassVar[list] = []
    calls: ClassVar[int] = 0
    declared: ClassVar[list] = []
    last_args: ClassVar[dict] = {}

    def __init__(self, url: str, timeout: float | None = None) -> None:
        self.url = url

    async def __aenter__(self):
        type(self).opened.append(self.url)
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def list_tools(self):
        return type(self).declared or [_Tool("searxng_web_search")]

    async def call_tool(self, name: str, args: dict):
        type(self).calls += 1
        type(self).last_args = dict(args or {})
        step = type(self).script.pop(0) if type(self).script else "результат поиска"
        if isinstance(step, Exception):
            raise step
        return _Result(step)


class _Result:
    """Ответ инструмента в форме клиента MCP: части с текстом."""

    def __init__(self, text: str) -> None:
        self.content = [_Part(text)]


class _Part:
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def fake_client(monkeypatch):
    """Подменить транспорт MCP и обнулить паузы: тест не должен спать по-настоящему."""
    import fastmcp

    _FakeClient.opened, _FakeClient.script, _FakeClient.calls = [], [], 0
    _FakeClient.declared, _FakeClient.last_args = [], {}
    monkeypatch.setattr(fastmcp, "Client", _FakeClient)
    monkeypatch.setattr(toolset, "MCP_RETRY_PAUSE_S", 0.0)
    return _FakeClient


async def _tool(box: McpToolbox, name: str = "localhost_3001_mcp_searxng_web_search"):
    return box.tools()[name].call


@pytest.mark.asyncio
async def test_one_session_serves_many_tool_calls(fake_client):
    """ГЛАВНОЕ свойство: сколько бы раз роль ни звала инструмент, инициализация ОДНА.

    Сессия на каждый вызов даёт 1 + N инициализаций на выпуск, и это упирается в
    лимит обёртки: «Too many requests» приходит от неё, а выглядит как отказ поиска.
    """
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp", name="searxng"),)) as box:
        call = await _tool(box)
        texts = [await call({"query": f"запрос {i}"}) for i in range(5)]

    assert _FakeClient.opened == ["http://localhost:3001/mcp"], \
        f"сессия открыта {len(_FakeClient.opened)} раз(а), а обязана быть одна на выпуск"
    assert len(texts) == 5 and all("результат поиска" in text for text in texts)


@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_succeeds(fake_client):
    """Лимит запросов — временная причина: повтор даёт результат вместо отказа выпуска."""
    _FakeClient.script = [RuntimeError("MCPError: Too many requests"), RuntimeError("timeout"), "нашлось"]
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp"),)) as box:
        text = await (await _tool(box))({"query": "ставка"})

    assert text == "нашлось", "временный отказ обязан дойти до результата повтором"
    assert _FakeClient.calls == 3
    assert _FakeClient.opened == ["http://localhost:3001/mcp"], "живую сессию переоткрывать незачем"


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried(fake_client):
    """Неизвестный инструмент не пройдёт и на третий раз: повторять его — тянуть время впустую."""
    _FakeClient.script = [ValueError("Unknown tool: searxng_web_search")]
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp"),)) as box:
        text = await (await _tool(box))({"query": "ставка"})

    assert text.startswith(SEARCH_FAILED), "наружу обязан идти маркер отказа поиска"
    assert _FakeClient.calls == 1, "постоянную ошибку повторять нельзя"


@pytest.mark.asyncio
async def test_broken_session_is_reopened(fake_client):
    """Мёртвую сессию переоткрываем: так одна сессия на выпуск не боится обрыва."""
    _FakeClient.script = [ConnectionError("connection closed"), "нашлось после переоткрытия"]
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp"),)) as box:
        text = await (await _tool(box))({"query": "ставка"})

    assert text == "нашлось после переоткрытия"
    assert _FakeClient.opened == ["http://localhost:3001/mcp"] * 2, "сессия обязана быть переоткрыта"


@pytest.mark.asyncio
async def test_failure_marker_is_never_emptiness(fake_client):
    """Отказ несёт маркер и причину: роль обязана отличить «поиск отказал» от «новостей нет»."""
    _FakeClient.script = [RuntimeError("timeout"), RuntimeError("timeout"), RuntimeError("timeout")]
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp"),)) as box:
        text = await (await _tool(box))({"query": "ставка"})

    assert text.startswith(f"{SEARCH_FAILED}: "), f"ожидался маркер, получено {text[:60]!r}"
    assert "timeout" in text, "причина обязана быть в тексте, иначе диагностика бесполезна"


@pytest.mark.asyncio
async def test_bad_address_is_skipped_without_session(monkeypatch):
    """Негодный адрес (IP вместо имени) не открывает сессию: инструментов нет, но и шума нет."""
    import fastmcp

    _FakeClient.opened = []
    monkeypatch.setattr(fastmcp, "Client", _FakeClient)
    async with McpToolbox((McpServer(url="http://127.0.0.1:3001/mcp"),)) as box:
        assert box.tools() == {}

    assert _FakeClient.opened == [], "адрес отсекается ДО подключения — это и есть смысл охранника"


def test_transient_and_broken_are_told_apart():
    """Классификация отказов — решение, от которого зависит «повторять или нет»."""
    assert _transient(RuntimeError("MCPError: Too many requests"))
    assert _transient(TimeoutError("timed out"))
    assert not _transient(ValueError("Unknown tool"))

    assert _looks_broken(ConnectionError("connection closed"))
    assert not _looks_broken(RuntimeError("Too many requests")), \
        "лимит не ломает сессию: переоткрывать её при лимите — тратить ещё одну инициализацию"


def test_searxng_settings_keeps_only_proven_engines():
    """Настройки нашего поиска: только движки, доказанные замером, и `bing` ВКЛЮЧЁН.

    Проверка структурная, зато ловит тихую поломку: в дефолтах SearXNG `bing` выключен, и если
    пропадёт строка `disabled: false`, движок молча исчезнет из выдачи — `keep_only` его оставит в
    списке, а результатов он не даст.
    """
    # Настройки инстанса поиска живут вне этого репозитория; в одиночном клоне их нет.
    import pytest
    import yaml
    from paths import WORKSPACE

    settings_path = WORKSPACE / "search/searxng/settings.yml"
    if not settings_path.exists():
        pytest.skip(f"нет настроек инстанса поиска ({settings_path}) — они живут вне этого репозитория")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    keep_only = settings["use_default_settings"]["engines"]["keep_only"]

    assert keep_only == ["google cse", "bing", "wikipedia"], \
        "набор движков — результат замера: brave/duckduckgo/startpage отвечали CAPTCHA и таймаутами"
    overrides = {engine["name"]: engine for engine in settings["engines"]}
    assert overrides["bing"]["disabled"] is False, "без этого bing останется выключенным по умолчанию"
    assert settings["search"]["formats"] == ["json"], "HTML-интерфейс нашему пути не нужен"
    assert settings["server"]["limiter"] is False, \
        "инстанс не публичный (loopback), а лимитер резал бы наши же запросы"


def test_declared_schema_drops_unsupported_search_fields():
    """Фильтры, которых наш поиск исполнить не может, из схемы УБИРАЮТСЯ — модель их не видит.

    Замер, ради которого это заведено: ОДИН И ТОТ ЖЕ запрос даёт **10** результатов без параметров и
    **0** — с `time_range=month`, потому что у `bing` объявлено `time_range_support: False`, а
    единственный движок с фильтром (`google cse`) висит `suspended`. Модель выбирала фильтр сама
    («свежие новости»), получала пусто, повторяла запросы и выжигала лимит кругов.
    """
    schema, stripped = toolset._declared_schema({
        "type": "object",
        "properties": {"query": {"type": "string"}, "time_range": {"type": "string"},
                       "language": {"type": "string"}},
        "required": ["query", "time_range"],
    })
    assert stripped is True
    assert set(schema["properties"]) == {"query"}, "фильтры обязаны исчезнуть из схемы"
    assert schema["required"] == ["query"], "и из списка обязательных тоже"


def test_declared_schema_leaves_foreign_tools_untouched():
    """Чужой инструмент, у которого этих полей нет, не трогаем — сужать возможности молча нельзя."""
    schema, stripped = toolset._declared_schema({"type": "object", "properties": {"url": {"type": "string"}}})
    assert stripped is False
    assert set(schema["properties"]) == {"url"}


def test_long_tool_answer_is_trimmed_with_an_explicit_mark():
    """Длинный ответ инструмента обрезается И помечается — иначе роль примет обрывок за полный текст.

    Зачем обрезка: прочитанная страница приходит целиком (замер: 3000 символов на статью), роль читает
    несколько подряд, а каждый круг отправляет всю историю — шаг упирался в лимит кругов, потратив
    45 995 токенов и не отдав строк `NEWS|`.
    """
    long_text = "а" * (toolset.MCP_RESULT_LIMIT_CHARS + 500)
    trimmed = toolset._mcp_result_text(_Result(long_text))
    assert len(trimmed) < len(long_text)
    assert trimmed.startswith("а" * 100)
    assert "обрезан" in trimmed, "обрезка обязана быть названа, а не молчаливой"

    short = toolset._mcp_result_text(_Result("короткий ответ"))
    assert short == "короткий ответ", "короткий ответ не трогаем"


@pytest.mark.asyncio
async def test_model_gets_search_free_of_filters_and_call_drops_them(fake_client):
    """Сквозная проверка: до модели — без фильтров, до поиска — без них же (защита в глубину)."""
    fake_client.declared = [_Tool("searxng_web_search", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "time_range": {"type": "string"},
                       "language": {"type": "string"}},
        "required": ["query"],
    })]
    async with McpToolbox((McpServer(url="http://localhost:3001/mcp"),)) as box:
        tool = next(iter(box.tools().values()))
        assert set(tool.parameters.get("properties", {})) == {"query"}
        assert "не поддерживаются" in tool.description, "модели надо сказать, почему фильтров нет"
        await tool.call({"query": "ставка", "time_range": "month", "language": "ru"})

    assert fake_client.last_args == {"query": "ставка"}, \
        "фильтры не должны дойти до поиска даже как аргументы вызова"


def test_long_json_answer_stays_parseable_when_trimmed():
    """JSON обрезается по СТРУКТУРЕ, а не по символам.

    Срез строки после сериализации даёт обрывок, который не разбирается ничем: у модели нет парсера,
    `llm._tool_answer` кладёт ответ в `content` как есть, поэтому половина данных тихо превращается в мусор.
    Поэтому список теряет ХВОСТОВЫЕ элементы, а причина остаётся видимой полем `_truncated`.
    """
    import json

    hits = [{"title": f"Новость {index}", "url": f"https://example.invalid/{index}", "text": "т" * 200}
            for index in range(40)]
    trimmed = toolset._mcp_result_text(_Result(json.dumps(hits, ensure_ascii=False)))

    payload = json.loads(trimmed)          #: главное утверждение: разбирается
    assert payload["_truncated"]["total"] == 40
    assert 0 < payload["_truncated"]["shown"] == len(payload["results"])
    assert len(trimmed) <= toolset.MCP_RESULT_LIMIT_CHARS
    assert payload["results"][0]["title"] == "Новость 0", "голова списка сохраняется: она и есть суть"


def test_long_json_object_names_what_was_cut():
    """Объект: урезанный список НАЗВАН, отброшенный крупный ключ ПЕРЕЧИСЛЕН.

    «_truncated: true» без подробностей заставляло бы модель гадать, что именно потеряно, — а решение
    «заметка об усечении обязана быть названной» в проекте уже принято для прозы.
    """
    import json

    hits = [{"title": f"Н{index}", "url": f"https://example.invalid/{index}", "text": "т" * 200}
            for index in range(40)]
    payload = {"query": "новости", "results": hits, "noise": "я" * 5000}
    trimmed = toolset._mcp_result_text(_Result(json.dumps(payload, ensure_ascii=False)))

    parsed = json.loads(trimmed)
    assert parsed["query"] == "новости", "полезный ключ не теряется"
    assert parsed["_truncated"]["dropped_keys"] == ["noise"], "отброшенный ключ обязан быть назван"
    assert parsed["_truncated"]["lists"]["results"]["total"] == 40
    assert 0 < len(parsed["results"]) < 40, "место, освобождённое сбросом ключа, заняли данные"
    assert len(trimmed) <= toolset.MCP_RESULT_LIMIT_CHARS


def test_pathological_json_is_named_not_mangled():
    """Ответ без списков (одна огромная строка) отдаёт ПОМЕТКУ, а не обрывок.

    Названная граница формата: показать кусок такой строки значило бы вернуть невалидный JSON, от которого
    мы и уходим. Модель узнаёт, что данные были и что они не влезли, — это правда, в отличие от мусора.
    """
    import json

    huge = {"page_text": "я" * 6000, "title": "заголовок"}
    trimmed = toolset._mcp_result_text(_Result(json.dumps(huge, ensure_ascii=False)))

    parsed = json.loads(trimmed)
    assert parsed["_truncated"]["dropped_keys"] == ["page_text"]
    assert parsed["title"] == "заголовок", "мелкое поле сохраняется: оно влезает"
    assert len(trimmed) <= toolset.MCP_RESULT_LIMIT_CHARS
