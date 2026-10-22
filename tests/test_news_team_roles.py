"""Состав команды: три роли, их инструменты и порядок — без фреймворка.

Состав задан спецификацией пункта, и проверять его надо отдельно от CrewAI: если проверка
живёт в тесте с фреймворком, то при его отсутствии перестаёт быть видно, из чего команда
состоит.
"""
from __future__ import annotations

import pytest

from news_agent.news_team.roles import AUTHOR, RESEARCHER, ROLES, VERIFIER, role
from news_agent.news_team.tools import ALL_TOOLS, BY_NAME


def test_team_has_exactly_three_roles_in_specified_order():
    """Порядок «исследователь → автор → проверяющий» — часть спецификации пункта."""
    assert [item.key for item in ROLES] == [RESEARCHER, AUTHOR, VERIFIER]


def test_researcher_and_verifier_have_tools_author_does_not():
    """Инструменты розданы по спецификации: поиск — исследователю, проверка и гейт — проверяющему."""
    assert role(RESEARCHER).tools == (BY_NAME["knowledge_search"].NAME,)
    # У проверяющего ОДИН инструмент — детерминированная проверка фактов. Гейт новости инструментом
    # НЕ даётся: иначе открыт путь спрятать новость в обход лестницы (роль вызывает гейт
    # сама и ставила `show:false`, а при деградации нашего вызова её слово подхватывалось как решение).
    # Исход гейта приходит роли данными, применяет его наш код.
    assert role(VERIFIER).tools == (BY_NAME["lost_facts"].NAME,)
    assert role(AUTHOR).tools == ()


def test_every_tool_in_roles_exists_in_the_registry():
    """Роль не может ссылаться на инструмент, которого нет: это опечатка, а не замысел."""
    for item in ROLES:
        for name in item.tools:
            assert name in BY_NAME, f"роль {item.key} ссылается на неизвестный инструмент {name!r}"


def test_unknown_role_is_a_programmer_error():
    with pytest.raises(ValueError):
        role("nonexistent")


def test_registry_lists_all_three_tools():
    assert sorted(module.NAME for module in ALL_TOOLS) == ["knowledge_search", "lost_facts", "news_gate"]
