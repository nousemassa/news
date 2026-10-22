"""Инструменты команды по отдельности — без фреймворка и без сети."""
from __future__ import annotations

import re

import pytest

from news_agent.news import facts_kept
from news_agent.news_team.tools import ALL_TOOLS
from news_agent.news_team.tools import facts as facts_tool
from news_agent.news_team.tools import retrieval as retrieval_tool


def test_tool_names_are_valid_function_names():
    """Имя инструмента уходит в API как имя функции, поэтому проверяется по правилу API.

    Тест появился из платного прогона: имена были русскими (`факты_потеряны`), и команда
    падала на шаге проверяющего с `OpenAI function name cannot be empty` — инструмент,
    который «есть» по структурной проверке, до модели не доходил вовсе. OpenAI-совместимый
    API принимает только `^[A-Za-z0-9_-]{1,64}$`, и проверить это можно бесплатно.
    """
    for module in ALL_TOOLS:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", module.NAME), (
            f"имя инструмента {module.NAME!r} не пройдёт как имя функции в API"
        )


class Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class Retriever:
    def __init__(self, chunks=None, error: Exception | None = None) -> None:
        self._chunks = chunks or []
        self._error = error
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 4):
        self.queries.append(query)
        if self._error:
            raise self._error
        return self._chunks


@pytest.mark.asyncio
async def test_retrieval_returns_fragment_texts():
    retriever = Retriever([Chunk("Задаток. Отличия от аванса"), Chunk("Дарение. Договор")])

    result = await retrieval_tool.search(retriever, "чем задаток отличается от аванса")

    assert "Задаток" in result and "Дарение" in result
    assert retriever.queries == ["чем задаток отличается от аванса"]


@pytest.mark.asyncio
async def test_retrieval_without_base_is_empty_not_an_error():
    """Нет базы знаний — это «знаний нет», а не сломанный прогон команды."""
    assert await retrieval_tool.search(None, "любой вопрос") == ""


@pytest.mark.asyncio
async def test_retrieval_does_not_raise_on_failure():
    """Недоступная база — пустой результат, причина в журнал: прогон продолжается."""
    result = await retrieval_tool.search(Retriever(error=RuntimeError("база недоступна")), "вопрос")

    assert result == ""


def test_facts_wrapper_matches_the_original_check():
    """Обёртка обязана давать ТОТ ЖЕ результат, что прямой вызов.

    `facts_kept` переписывать нельзя; этот тест и есть защита от того,
    что обёртка начнёт судить о потере фактов по своим критериям, а публикация — по
    другим.
    """
    original = "Цена выросла на 12% и составила 8 900 000 ₽ с 1 января 2026 года."
    shortened = "Цена выросла и составила 8 900 000 ₽."

    assert facts_tool.missing_facts(original, shortened) == facts_kept(original, shortened)
    assert facts_tool.missing_facts(original, original) == []
