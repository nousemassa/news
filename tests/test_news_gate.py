"""Гейт новости — инструмент проверяющего и правило применения решения.

Без сети и без моделей: решатель подставляется двойником, поэтому здесь проверяется ровно то, что
принадлежит нам — деградация, чтение уровня и КТО применяет решение.
"""
from __future__ import annotations

import pytest

from agent_core.decisions.client import Decision, SolverUnavailable
from agent_core.decisions.ladder import LEVEL_CLOUD, LEVEL_SOLVER, REASON_UNAVAILABLE
from news_agent.news_team.tools import news_gate


def awaiting(answer: str, probability: float):
    """Двойник решателя: отвечает заданным вариантом с заданной уверенностью.

    Обычная функция, возвращающая асинхронную: `async def` вернул бы корутину, а не функцию, и
    вызов упал бы — причём так, что это было бы похоже на недоступность решателя.
    """
    async def ask(state: str, question: str, options: dict) -> Decision:
        assert question == news_gate.QUESTION, "вопрос гейта один на инструмент и на замер"
        assert set(options) == set(news_gate.OPTIONS), "варианты те же, что в контракте гейта"
        return Decision(answer=answer, top_probability=probability, probabilities={answer: probability})

    return ask


async def broken(state: str, question: str, options: dict) -> Decision:
    raise SolverUnavailable("соединение отклонено (тест)")


@pytest.mark.asyncio
async def test_confident_answer_closes_the_case_locally():
    step = await news_gate.ask_gate("текст новости", ask=awaiting("да", 0.95), threshold=0.9)

    assert step.level == LEVEL_SOLVER
    assert step.closed_locally is True


@pytest.mark.asyncio
async def test_unsure_answer_escalates_to_cloud():
    step = await news_gate.ask_gate("текст новости", ask=awaiting("нет", 0.6), threshold=0.9)

    assert step.level == LEVEL_CLOUD
    assert step.closed_locally is False


@pytest.mark.asyncio
async def test_missing_solver_is_degradation_not_failure():
    """Решателя нет — случай уходит на уровень 3, а прогон продолжается.

    Исключение здесь было бы хуже отказа: роль получила бы техническую ошибку вместо «не знаю»,
    а прогон — повод упасть из-за локального сервиса.
    """
    step = await news_gate.ask_gate("текст новости", ask=broken, threshold=0.9)

    assert step.level == LEVEL_CLOUD
    assert step.reason == REASON_UNAVAILABLE


def test_only_a_confident_local_no_rejects_the_news():
    """Отклоняет ровно один исход: уверенное локальное «нет». Всё остальное — показываем."""
    from agent_core.decisions.ladder import choose

    assert news_gate.rejects(choose(Decision("нет", 0.95), threshold=0.9)) is True
    assert news_gate.rejects(choose(Decision("нет", 0.60), threshold=0.9)) is False, "неуверенное «нет» не скрывает"
    assert news_gate.rejects(choose(Decision("да", 0.95), threshold=0.9)) is False
    assert news_gate.rejects(choose(None, threshold=0.9)) is False, "без решателя показываем"


def test_the_application_of_the_decision_is_ours_not_the_roles():
    """Решение применяет НАШ код: уверенное локальное «нет» отменяет показ, что бы ни написала роль.

    Документы пункта называют это красным флагом («применяет решение всё равно мост»), и это же
    делает гейт настоящим: иначе уверенное дешёвое решение перебивалось бы текстом дорогого уровня.
    """
    from agent_core.decisions.ladder import choose

    confident_no = choose(Decision("нет", 0.95), threshold=0.9)

    assert news_gate.apply(confident_no, claimed_show=True) is False
    assert news_gate.apply(confident_no, claimed_show=False) is False


def test_unsure_gate_leaves_the_decision_to_the_role():
    """Неуверенность гейта и его отсутствие НЕ скрывают новость: показ — состояние по умолчанию."""
    from agent_core.decisions.ladder import choose

    unsure = choose(Decision("нет", 0.60), threshold=0.9)

    assert news_gate.apply(unsure, claimed_show=True) is True
    assert news_gate.apply(unsure, claimed_show=False) is False, "роль вправе отклонить сама"


def test_the_verdict_names_the_level_so_savings_can_be_counted():
    """В словах для роли назван УРОВЕНЬ, а не только ответ: без этого экономию не посчитать."""
    from agent_core.decisions.ladder import choose

    local = news_gate.format_for_model(choose(Decision("да", 0.95), threshold=0.9))
    cloud = news_gate.format_for_model(choose(None, threshold=0.9))

    assert "Уровень лестницы: 2" in local and "облако не привлекалось" in local
    assert "Уровень лестницы: 3" in cloud
