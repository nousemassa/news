"""Замер лестницы: отбор размеченных примеров и три числа.

Проверяется на двойнике решателя — то есть без сети и без модели. Замер обязан быть воспроизводимым
на любой машине: он же печатает числа приёмки.
"""
from __future__ import annotations

import pytest

from agent_core.decisions.client import Decision
from agent_core.decisions.ladder import LEVEL_CLOUD, LEVEL_SOLVER
from news_agent.decisions.labels import (
    LABEL_HIDE,
    LABEL_SHOW,
    LABELS,
    ORIGIN_FINDINGS,
    ORIGIN_HAND,
    ORIGIN_OFFTOPIC,
    expected_answer,
    normalise_label,
)
from news_agent.decisions.report import labelled, measure, report


def test_labels_are_numbers_as_the_owner_decided():
    """Метка — число: 1 показывать, 0 не показывать. Слова в наборе больше не используются."""
    assert (LABEL_SHOW, LABEL_HIDE) == ("1", "0")
    assert expected_answer(LABEL_SHOW) == "да"
    assert expected_answer(LABEL_HIDE) == "нет"


def test_both_spellings_of_a_number_label_are_the_same():
    """В YAML `label: 0` — целое, `label: "0"` — строка. Разметка владельца должна приниматься в обеих.

    Иначе половина набора молча выпала бы как «не размечено», и число выглядело бы посчитанным.
    """
    assert normalise_label(0) == "0"
    assert normalise_label("0") == "0"
    assert normalise_label(1) == "1"
    assert normalise_label(" 1 ") == "1"


def test_boolean_is_not_a_label():
    """`label: true` в YAML — не метка: bool в Python подкласс int, и без проверки он дал бы «1»."""
    assert normalise_label(True) == ""
    assert normalise_label(False) == ""


def test_only_labelled_items_with_text_are_measured():
    """«Не размечено» и «непонятно» считаются РАЗДЕЛЬНО: первое — работа, второе — ошибка."""
    items = [
        {"id": 1, "text": "новость", "label": 1},            # целое из YAML
        {"id": 2, "text": "новость", "label": ""},           # ещё не размечено
        {"id": 3, "text": "   ", "label": LABEL_HIDE},         # нечего оценивать
        {"id": 4, "text": "новость", "label": "может быть"},  # метка вне списка
    ]

    ready, unlabeled, invalid, empty_text = labelled(items)

    assert [item["id"] for item in ready] == [1]
    assert invalid == [4], "битая метка названа номером, а не спрятана в «пропущено»"
    assert empty_text == [3], "«метка есть, текста нет» — ошибка данных, а не «не размечено»"
    assert unlabeled == 1


def solver(answer_by_text: dict[str, tuple[str, float]]):
    """Двойник решателя: ответ и уверенность берутся по тексту примера."""
    async def ask(state: str, question: str, options: dict) -> Decision:
        answer, probability = answer_by_text[state]
        return Decision(answer=answer, top_probability=probability, probabilities={answer: probability})

    return ask


@pytest.mark.asyncio
async def test_three_numbers_are_counted_separately():
    """Три числа: закрыто локально, точность на закрытых, точность по всем ответам.

    Смешивать их нельзя: без второго «закрыли 100 %» выглядело бы победой даже если решения
    угаданы наугад.
    """
    items = [
        {"id": 1, "text": "А", "label": LABEL_SHOW},   # уверенно «да» — верно, локально
        {"id": 2, "text": "Б", "label": LABEL_HIDE},    # уверенно «да» — ОШИБКА, но локально
        {"id": 3, "text": "В", "label": LABEL_HIDE},    # неуверенно «нет» — уходит в облако
    ]

    result = await measure(items, threshold=0.9,
                           ask=solver({"А": ("да", 0.95), "Б": ("да", 0.95), "В": ("нет", 0.5)}))

    assert result["items"] == 3
    assert result["closed"] == 2
    assert result["closed_share"] == pytest.approx(2 / 3)
    assert result["accuracy_on_closed"] == pytest.approx(0.5), "одна из двух локальных ошибка"
    # Ответ МОДЕЛИ по всем примерам: А верно, Б неверно, В верно («нет» при метке «не показывать»).
    # Уход В в облако НЕ делает её ответ неверным: уход — это решение спросить облако, а не ошибка.
    assert result["accuracy_raw"] == pytest.approx(2 / 3)
    assert result["false_rejects"] == 0, "важное не скрывалось"
    assert result["false_accepts"] == 1, "показал то, что владелец отклонил — мягкая ошибка"
    assert result["escalated_reasons"]


@pytest.mark.asyncio
async def test_unavailable_solver_escalates_everything_and_closes_nothing():
    async def broken(state: str, question: str, options: dict):
        from agent_core.decisions.client import SolverUnavailable

        raise SolverUnavailable("нет сервиса (тест)")

    result = await measure([{"id": 1, "text": "А", "label": LABEL_SHOW}], threshold=0.9, ask=broken)

    assert result["closed"] == 0
    assert result["closed_share"] == 0.0
    assert result["threshold"] == 0.9


def test_report_says_it_cannot_count_without_labels(capsys):
    """Пустой набор — «числа нет», а не нули: ноль выглядел бы измеренным результатом."""
    report({"items": 0, "closed": 0, "closed_share": 0.0, "accuracy_on_closed": 0.0,
            "accuracy_raw": 0.0, "escalated_reasons": {}, "threshold": 0.9, "rows": []}, unlabeled=0)

    assert "число получить нельзя" in capsys.readouterr().out


def test_report_prints_the_levels_meaning(capsys):
    report({"items": 2, "closed": 1, "closed_share": 0.5, "accuracy_on_closed": 1.0,
            "accuracy_raw": 0.5, "escalated_reasons": {"уверенность ниже порога": 1},
            "threshold": 0.9, "rows": []}, unlabeled=1)

    out = capsys.readouterr().out
    assert "закрыто БЕЗ облака" in out
    assert "качество экономии" in out
    assert "уверенность ниже порога" in out


def test_levels_used_by_the_report_match_the_ladder():
    """Число закрытых считается по уровню лестницы, а не по догадке в отчёте."""
    assert LEVEL_SOLVER != LEVEL_CLOUD


def test_report_names_the_examples_with_a_broken_label(capsys):
    """Битые метки печатаются номерами: иначе пример выпал бы незаметно."""
    report({"items": 1, "closed": 1, "closed_share": 1.0, "accuracy_on_closed": 1.0,
            "accuracy_raw": 1.0, "escalated_reasons": {}, "threshold": 0.9, "rows": []},
           unlabeled=0, invalid=[7, 8])

    out = capsys.readouterr().out
    assert "[7, 8]" in out and "НЕ в замере" in out


# --- охранник самого набора -------------------------------------------------------------
# Набор — коммитимый артефакт, и правка текста внутри него может сломать YAML незаметно для тестов:
# разбор набора проверяется отдельно от запуска отчёта, на котором поломка иначе всплывает.

def shipped_set_path():
    """Путь к набору меток. Набор живёт вне этого репозитория (в воркспейсе), поэтому его может не быть."""
    from paths import WORKSPACE

    return WORKSPACE / "eval" / "decisions" / "labeled.yaml"


def _require_set():
    import pytest

    path = shipped_set_path()
    if not path.exists():
        pytest.skip(f"набора меток нет ({path}) — он живёт вне этого репозитория")
    return path


def test_the_shipped_set_parses_and_has_only_valid_labels():
    """Набор читается, метки допустимы, а инструкция в нём называет ТУ ЖЕ форму метки, что код."""
    import yaml

    document = yaml.safe_load(_require_set().read_text(encoding="utf-8"))

    assert isinstance(document, dict), "набор должен быть объектом YAML"
    items = document.get("items")
    assert isinstance(items, list) and items, "в наборе должны быть примеры"
    for item in items:
        assert str(item.get("text") or "").strip(), f"пустой текст у примера {item.get('id')}"
        assert normalise_label(item.get("label")) in ("", *LABELS), (
            f"метка примера {item.get('id')} недопустима: {item.get('label')!r}"
        )
    note = str(document.get("note") or "")
    assert "label = 1" in note and "label = 0" in note, (
        "инструкция внутри набора разошлась с кодом: владелец прочитает одно, а примется другое"
    )


# --- блоки набора: находки / отклоняемые / ручные ---------------------------------------

@pytest.mark.asyncio
async def test_numbers_are_split_by_block():
    """Блоки описывают разные распределения: среднее по ним ничего не говорит о продакшене."""
    items = [
        {"id": 1, "text": "А", "label": LABEL_SHOW, "origin": ORIGIN_FINDINGS},
        {"id": 2, "text": "Б", "label": LABEL_HIDE, "origin": ORIGIN_OFFTOPIC},
        {"id": 3, "text": "В", "label": LABEL_HIDE, "origin": ORIGIN_HAND},
    ]
    ask = solver({"А": ("да", 0.95), "Б": ("нет", 0.95), "В": ("да", 0.95)})  # на ручном ошибся

    result = await measure(items, threshold=0.9, ask=ask)

    blocks = result["by_origin"]
    assert set(blocks) == {ORIGIN_FINDINGS, ORIGIN_OFFTOPIC, ORIGIN_HAND}
    assert blocks[ORIGIN_FINDINGS]["accuracy_raw"] == 1.0
    assert blocks[ORIGIN_OFFTOPIC]["accuracy_raw"] == 1.0
    assert blocks[ORIGIN_HAND]["accuracy_raw"] == 0.0, "ошибка на ручном примере видна в своём блоке"
    assert result["accuracy_raw"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_missing_origin_counts_as_findings():
    """Старые записи набора писались без `origin` — они и есть находки, а не «неизвестно что»."""
    result = await measure([{"id": 1, "text": "А", "label": LABEL_SHOW}], threshold=0.9,
                           ask=solver({"А": ("да", 0.95)}))

    assert set(result["by_origin"]) == {ORIGIN_FINDINGS}


def test_report_warns_when_a_block_has_only_one_class(capsys):
    """Однородный блок — «точность» в нём ничего не различает: об этом надо сказать вслух."""
    report({"items": 2, "closed": 2, "closed_share": 1.0, "accuracy_on_closed": 1.0,
            "accuracy_raw": 1.0, "escalated_reasons": {}, "threshold": 0.9, "rows": [],
            "by_origin": {ORIGIN_FINDINGS: {"items": 2, "labels": {LABEL_SHOW: 2},
                                            "closed": 2, "closed_share": 1.0,
                                            "accuracy_on_closed": 1.0, "accuracy_raw": 1.0}}},
           unlabeled=0)

    out = capsys.readouterr().out
    assert "по блокам набора" in out
    assert "не различает" in out


def test_report_stays_quiet_when_both_classes_are_present(capsys):
    report({"items": 2, "closed": 2, "closed_share": 1.0, "accuracy_on_closed": 1.0,
            "accuracy_raw": 1.0, "escalated_reasons": {}, "threshold": 0.9, "rows": [],
            "by_origin": {ORIGIN_OFFTOPIC: {"items": 2, "labels": {LABEL_SHOW: 1, LABEL_HIDE: 1},
                                            "closed": 2, "closed_share": 1.0,
                                            "accuracy_on_closed": 1.0, "accuracy_raw": 1.0}}},
           unlabeled=0)

    assert "не различает" not in capsys.readouterr().out


def test_shipped_set_note_matches_the_single_source():
    """Инструкция в наборе — РОВНО та, что в `labels.SET_NOTE`: иначе владелец размечает по одному
    критерию, а точность считается по другому."""
    import yaml

    from news_agent.decisions.labels import SET_NOTE

    document = yaml.safe_load(_require_set().read_text(encoding="utf-8"))

    assert document["note"].strip() == SET_NOTE.strip(), "инструкция в наборе разошлась с кодом"


@pytest.mark.asyncio
async def test_escalation_with_a_right_answer_is_not_counted_as_an_error():
    """Дефект метрики, найденный на реальном замере: уход в облако считался ошибкой.

    При росте порога «точность» падала, хотя решатель не менялся, — потому что у ушедших случаев ответ
    не переносится (`answer` пуст у уровня 3) и сравнение давало «неверно». Ошибка ответа и решение
    спросить облако — разные вещи, и мерить их одним числом нельзя.
    """
    items = [
        {"id": 1, "text": "А", "label": LABEL_HIDE},  # модель ответила «нет» верно, но неуверенно
        {"id": 2, "text": "Б", "label": LABEL_HIDE},  # модель ответила «да» неверно, уверенно
    ]

    result = await measure(items, threshold=0.9,
                           ask=solver({"А": ("нет", 0.5), "Б": ("да", 0.95)}))

    assert result["closed"] == 1 and result["items"] == 2
    assert result["accuracy_raw"] == pytest.approx(0.5), "у А ответ ВЕРНЫЙ, у Б — нет"
    assert result["accuracy_on_closed"] == 0.0, "у закрытого локально ответ неверный"


# --- метки не должны теряться при повторном сборе ---------------------------------------------

def test_rewriting_the_set_carries_labels_over():
    """Сборщик накопительный: повторный прогон обязан СОХРАНИТЬ разметку.

    Сборщик, пишущий всем пустую метку, стирает разметку владельца одним запуском — а на ней стоят
    все числа замера.
    """
    from news_agent.decisions.labels import carry_labels

    previous = [{"url": "https://a/1", "title": "Первая", "label": "1"},
                {"url": "", "title": "Вторая новость", "label": 0}]
    current = [{"url": "https://a/1", "title": "Первая", "label": ""},
               {"url": "", "title": "Вторая новость", "label": ""},
               {"url": "https://a/3", "title": "Третья", "label": ""}]

    result = carry_labels(previous, current)

    assert [item["label"] for item in result] == ["1", "0", ""], "метки перенесены, новая пустая"


def test_matching_works_by_title_when_a_url_appears_later():
    """У находок команды адрес мог появиться после разметки — тогда метка ищется по заголовку."""
    from news_agent.decisions.labels import carry_labels

    previous = [{"url": "", "title": "Продажи новостроек выросли", "label": "0"}]
    current = [{"url": "https://rbc/1", "title": "Продажи новостроек выросли", "label": ""}]

    assert carry_labels(previous, current)[0]["label"] == "0"


def test_explicit_reset_clears_labels():
    """Стереть разметку можно ТОЛЬКО явно: потеря должна быть решением, а не следствием запуска."""
    from news_agent.decisions.labels import carry_labels

    previous = [{"url": "https://a/1", "title": "Первая", "label": "1"}]
    current = [{"url": "https://a/1", "title": "Первая", "label": "1"}]

    assert carry_labels(previous, current, reset=True)[0]["label"] == ""


def test_unknown_origin_is_refused_not_silently_treated_as_findings():
    """Блок влияет на чтение числа: без проверки пример молча попал бы в «находки» (распределение прода)."""
    from news_agent.decisions.report import check_origins

    items = [{"id": 1, "origin": "findings"}, {"id": 2, "origin": "опечатка"}, {"id": 3, "origin": ""}]

    assert check_origins(items) == [2], "пустой блок допустим (старые записи), опечатка — нет"
