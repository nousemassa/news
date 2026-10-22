"""Разбор вердикта проверяющего — без фреймворка и без сети.

Живой прогон показал главный риск: проверяющий возвращает ОТЧЁТ о проверке вместо заметки, и это
уехало бы в публикацию. Поэтому выход роли — служебная строка, а её отсутствие — отказ шага.
"""
from __future__ import annotations

import pytest

from news_agent.news_team.verdict import VerdictError, parse_verdict, strip_verdict


def test_verdict_is_parsed_into_text_and_lost_facts():
    text, lost, show = parse_verdict(
        'VERDICT|{"text":"Ставка выросла на 2%.","lost":["18%"]}'
    )

    assert text == "Ставка выросла на 2%."
    assert lost == ("18%",)
    assert show is True, "показ — состояние по умолчанию: скрывать можно только явно"


def test_empty_lost_means_no_losses():
    text, lost, show = parse_verdict('VERDICT|{"text":"Заметка.","lost":[]}')

    assert text == "Заметка."
    assert lost == ()
    assert show is True


def test_answer_without_marker_is_a_failure_not_an_empty_release():
    """Отчёт о проверке — это НЕ заметка: шаг обязан провалиться, а не уехать в публикацию."""
    with pytest.raises(VerdictError):
        parse_verdict("Инструмент вернул пустой список — потерь нет. Проверка показала, что…")


def test_broken_json_is_a_failure():
    with pytest.raises(VerdictError):
        parse_verdict("VERDICT|{text: нет кавычек}")


def test_verdict_without_text_is_a_failure():
    with pytest.raises(VerdictError):
        parse_verdict('VERDICT|{"text":"   ","lost":[]}')


def test_text_taking_is_literal():
    """Вердикт берём целиком из поля text: пояснения рядом с маркером в публикацию не едут."""
    text, _, _ = parse_verdict(
        "Вот мой вердикт:\n" 'VERDICT|{"text":"Цены выросли на 12%.","lost":[]}' "\nСпасибо."
    )

    assert text == "Цены выросли на 12%."


def test_strip_removes_the_service_line():
    assert strip_verdict('текст\nVERDICT|{"text":"x","lost":[]}') == "текст"


def test_verdict_can_hide_the_news():
    """Гейт новости закрыл вопрос локально — роль ставит show: false."""
    _, _, show = parse_verdict('VERDICT|{"text":"Текст.","lost":[],"show":false}')

    assert show is False


def test_verdict_with_non_boolean_show_is_a_failure():
    """`show: "нет"` — не булево: догадываться о смысле нельзя, шаг не состоялся."""
    with pytest.raises(VerdictError):
        parse_verdict('VERDICT|{"text":"Текст.","lost":[],"show":"нет"}')
