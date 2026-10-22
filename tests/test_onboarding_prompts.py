"""Промпт агента новостей в репозитории обязан держать контракт парсера.

Зачем тест, а не доверие к тексту промпта. `agents_md` агента — ОТДЕЛЬНЫЙ источник формата, и он
пересиливает текст хода: «3–5 пунктов, до 12 строк» вместо строк `NEWS|{json}` даёт прозу, которую
`parse_news_blocks` отбрасывает, — и наружу уходит
«ничего не нашёл» вместо новостей. Строка промпта — тоже контракт, поэтому он закреплён ЗДЕСЬ:
файл в репозитории обязан требовать `NEWS|` и `NEWS_NONE` и не возвращать отменённый формат.
"""
from __future__ import annotations

from paths import PROJECT

from news_agent import news

PROMPT_FILE = PROJECT / "onboarding" / "news-hunter.md"
#: Формат, отменённый правкой: если он вернётся в промпт, агент снова начнёт отвечать прозой.
RETIRED_FORMAT = "3–5 пунктов"


def test_prompt_file_is_the_repo_source_of_truth():
    assert PROMPT_FILE.is_file(), f"нет файла промпта {PROMPT_FILE} — репозиторий источник, платформа цель выката"


def test_prompt_requires_the_parser_contract():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert news.NEWS_PREFIX in text, "промпт обязан требовать строки NEWS|"
    assert news.NEWS_NONE in text, "промпт обязан называть ответ «ничего не найдено»"


def test_prompt_does_not_prescribe_the_old_format():
    """Отменённый формат можно НАЗЫВАТЬ (чтобы отменить) — но не предписывать.

    Проверка построчная нарочно: сам факт упоминания ничего не значит (в промпте есть строка
    «Ранее заданный формат … ОТМЕНЁН»), а вот строка, которая его ТРЕБУЕТ, — дефект: тогда агент
    снова ответит прозой, и `parse_news_blocks` её отбросит.
    """
    for number, line in enumerate(PROMPT_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if RETIRED_FORMAT in line:
            assert "ОТМЕНЁН" in line, (
                f"{PROMPT_FILE.name}:{number}: промпт снова предписывает отменённый формат: {line!r}")


def test_prompt_defers_the_format_to_the_turn():
    """Формат задаёт ХОД, а не промпт агента — иначе два источника формата спорят (это и был дефект)."""
    assert "КАЖДОМ ходе" in PROMPT_FILE.read_text(encoding="utf-8")


def test_prompt_distinguishes_news_lines_from_plain_text():
    """Строки `NEWS|` — только когда ход их ТРЕБУЕТ; иначе (переписать новость) — обычный текст.

    Зачем именно так. Ход «Переписать» УПОМИНАЕТ `NEWS|` — в запрете («без строк NEWS|»). Формулировка
    «если в ходе задан формат NEWS|» читалась бы буквально и на этом ходу, и агент вернул бы строку
    вместо текста, сломав правку карточки. Поэтому условие — «ход ТРЕБУЕТ», и обе ветки названы явно.
    """
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "ТРЕБУЕТ" in text, "условие должно требовать, а не просто упоминать формат"
    assert "обычный текст" in text, "должна быть явно названа ветка «текст вместо строк»"


def test_turn_text_and_prompt_agree_on_one_format():
    """Текст хода и промпт агента задают ОДИН формат — иначе агент выбирает сам."""
    turn = news.build_news_turn(topic="ипотечные ставки", max_items=5)
    assert news.NEWS_PREFIX in turn and news.NEWS_NONE in turn
