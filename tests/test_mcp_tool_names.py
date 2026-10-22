"""Разбор адресов MCP-серверов команды — без сети и без внешних зависимостей.

Правило имени инструмента зеркалит адаптер фреймворка, но сверить наш расчёт с фактическим кодом
фреймворка НЕЧЕМ: вечно пропускаемый тест сравнения выглядит проверкой, не будучи ею, поэтому его
нет. Источник правила записан в докстроке `news_agent.news_team.tools.mcp`; эталоном служат наши
примеры и живые вызовы.

Файл появился после живой проверки: адрес по IP давал имя инструмента, начинающееся с цифры, и
OpenAI-совместимый API его отвергал — то есть исследователь оставался без поиска в интернете.
Здесь проверяется правило, по которому такой адрес отсекается ДО прогона, и то, что отсев не
молчит.

Второй дефект того же класса (русские имена инструментов) виден в примерах ниже: санитайзер
вырезает не-ASCII, и от русского имени остаётся пустая строка.
"""
from __future__ import annotations

import logging

import pytest

from news_agent.news_team.tools.mcp import (
    FUNCTION_NAME_RE,
    address_problem,
    final_tool_name,
    parse_servers,
    sanitized_name,
    tool_name_prefix,
)


def test_ip_address_is_rejected_with_a_reason(caplog):
    """Адрес по IP негоден, и причина названа словами — а не «неверный формат»."""
    with caplog.at_level(logging.WARNING):
        servers = parse_servers("searxng=http://127.0.0.1:3000/mcp")

    assert servers == ()
    assert "имя функции" in caplog.text
    assert "127.0.0.1" in caplog.text


def test_hostname_address_is_accepted():
    servers = parse_servers("searxng=http://localhost:3000/mcp")

    assert [s.url for s in servers] == ["http://localhost:3000/mcp"]
    assert servers[0].name == "searxng"


def test_prefix_follows_the_framework_rule():
    assert tool_name_prefix("http://localhost:3000/mcp") == "localhost:3000_mcp"


def test_ip_prefix_starts_with_a_digit_and_hostname_does_not():
    assert tool_name_prefix("http://127.0.0.1:3000/mcp").startswith("127")
    assert tool_name_prefix("http://localhost:3000/mcp")[0].isalpha()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # двоеточие и слэши становятся подчёркиваниями — как у санитайзера фреймворка
        ("http://localhost:3000/mcp", "localhost_3000_mcp_searxng_web_search"),
        ("http://127.0.0.1:3000/mcp", "127_0_0_1_3000_mcp_searxng_web_search"),
    ],
)
def test_final_tool_name_follows_the_api_rule(url, expected):
    """Имя собирается по правилу санитайзера: недопустимые символы в `_`, схлопывание, не длиннее 64."""
    assert final_tool_name(url) == expected


def test_only_the_hostname_address_passes_the_api_rule():
    assert FUNCTION_NAME_RE.fullmatch(final_tool_name("http://localhost:3000/mcp"))
    assert not FUNCTION_NAME_RE.fullmatch(final_tool_name("http://127.0.0.1:3000/mcp"))


def test_sanitizer_drops_non_ascii_like_the_framework():
    """Русское имя инструмента вырезается в пустое — API такое имя не примет."""
    assert sanitized_name("факты_потеряны") == ""


def test_address_in_the_example_is_usable():
    """Адрес из `.env.example` обязан проходить охранник: иначе пользователь скопирует нерабочее.

    Проверять свойство на «памятке» в коде (`default_servers`) значило бы, что код проверяет сам
    себя, а настоящий адрес живёт в настройке и может разойтись с правилом незаметно. Литералов в
    коде нет: адрес ровно один — в `.env`.

    `TEAM_MCP_SERVERS` принадлежит НОВОСТЯМ, поэтому пример читаем в `.env.example` этого проекта.
    """

    from paths import PROJECT

    example = (PROJECT / ".env.example").read_text(encoding="utf-8")
    raw = next(line.split("=", 1)[1].strip() for line in example.splitlines()
               if line.startswith("TEAM_MCP_SERVERS="))

    servers = parse_servers(raw)
    assert servers, f"адрес из примера не разобрался: {raw!r}"
    for server in servers:
        assert address_problem(server.url) == "", server.url


def test_empty_setting_means_no_servers():
    """Пусто — законное состояние: команда работает своими инструментами (деградация)."""
    assert parse_servers("") == ()


@pytest.mark.parametrize("raw", ["не-адрес", "ftp://localhost/x", "http://"])
def test_malformed_addresses_are_skipped(raw):
    assert parse_servers(raw) == ()
