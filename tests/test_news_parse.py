"""Контракт и разбор ответа новостного агента: строки NEWS|, валидация, дедуп, длина."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from news_agent import news

MSK = ZoneInfo("Europe/Moscow")


def _line(title: str = "Заголовок", text: str = "Т" * 60, url: str = "https://example.invalid/a",
          source: str = "example") -> str:
    payload = {"title": title, "text": text, "url": url, "source": source}
    return news.NEWS_PREFIX + json.dumps(payload, ensure_ascii=False)


def test_three_valid_lines_to_three_cards():
    items, status = news.parse_news_blocks("\n".join([_line(url="https://a.invalid/1"),
                                                      _line(url="https://a.invalid/2"),
                                                      _line(url="https://a.invalid/3")]), max_items=5)
    assert status == "ok"
    assert [item.url for item in items] == ["https://a.invalid/1", "https://a.invalid/2", "https://a.invalid/3"]


def test_broken_and_foreign_lines_are_skipped():
    raw = "\n".join([
        "Вот новости:",                               # проза — игнор
        "NEWS|{не json}",                             # битая строка — игнор
        'NEWS|{"title":"","text":"короткий","url":"https://a.invalid/0"}',   # пустой заголовок
        'NEWS|{"title":"ок","text":"кор", "url":"https://a.invalid/1"}',      # текст короткий
        'NEWS|{"title":"ок","text":"' + "т" * 60 + '","url":"ftp://a.invalid/2"}',  # не http(s)
        _line(url="https://a.invalid/3"),
    ])
    items, status = news.parse_news_blocks(raw, max_items=5)
    assert status == "ok"
    assert [item.url for item in items] == ["https://a.invalid/3"]


def test_duplicates_by_url_are_dropped():
    items, _ = news.parse_news_blocks(f"{_line()}\n{_line()}", max_items=5)
    assert len(items) == 1


def test_news_none_and_empty_statuses():
    assert news.parse_news_blocks("NEWS_NONE", max_items=5) == ([], "none")
    assert news.parse_news_blocks("", max_items=5) == ([], "empty")
    # Проза — НЕ «новостей нет»: это нарушенный контракт роли, и владельцу его надо назвать иначе.
    # Прежде оба состояния звались `none`, и сообщение «агент не отдал карточек» врало о причине.
    assert news.parse_news_blocks("Агент написал прозу без строк", max_items=5) == ([], "format")


def test_status_names_the_reason_not_just_success_or_failure():
    """Статус различает ОТКАЗ ИСТОЧНИКА, пустую выдачу и плохой формат — три разных лечения."""
    assert news.parse_news_blocks(f"{news.SEARCH_FAILED}: searxng_web_search не сработал", max_items=5) \
        == ([], "search_failed")
    assert news.parse_news_blocks(news.NEWS_NONE, max_items=5) == ([], "none")
    assert news.parse_news_blocks("Заголовок. Текст без служебных строк.", max_items=5) == ([], "format")


def test_cap_and_card_limit():
    raw = "\n".join([_line(url=f"https://a.invalid/{i}") for i in range(9)])
    items, _ = news.parse_news_blocks(raw, max_items=5)
    assert len(items) == 5, "потолок новостей за прогон"

    long_line = _line(text="т" * 5000)
    items, _ = news.parse_news_blocks(long_line, max_items=5, card_limit=3500)
    assert len(items) == 1
    assert len(items[0].title) + len(items[0].text) <= 3500
    assert items[0].text.endswith("…")


def test_strip_and_neutralize():
    raw = "Текст\n" + _line() + "\nещё текст"
    assert news.NEWS_PREFIX not in news.strip_news_blocks(raw)
    assert news.strip_news_blocks(raw).startswith("Текст")
    assert "NEWS·|" in news.neutralize_news_blocks(_line()), "ввод владельца не должен давать наших строк"


def test_card_and_publish_text():
    items, _ = news.parse_news_blocks(_line(source="Источник-имя"), max_items=1)
    item = items[0]
    card = news.format_card(item, tz=MSK, found_at=datetime(2026, 10, 8, 11, 30, tzinfo=UTC))
    assert "📰 Заголовок" in card and "🕒 найдено 08.10 14:30" in card and item.url in card
    published = news.publish_text(item)
    assert published.endswith(f"Источник: {item.url}")
    assert news.publish_text(item).count("Источник:") == 1, "источник добавляется один раз"


def test_utf16_len_counts_emoji_twice():
    assert news.utf16_len("abc") == 3
    assert news.utf16_len("📰") == 2, "эмодзи занимает две единицы UTF-16 — иначе Telegram обрежет"


def test_chunking_respects_limit_and_paragraphs():
    text = "\n\n".join("абзац " + "т" * 900 for _ in range(6))
    chunks = news.chunk_for_telegram(text, limit=1000)
    assert len(chunks) > 1
    assert all(news.utf16_len(chunk) <= 1000 for chunk in chunks), "ни один кусок не длиннее лимита"
    assert all(chunk.strip() for chunk in chunks)
    assert news.chunk_for_telegram("коротко", limit=1000) == ["коротко"]


def test_build_news_turn_contains_contract_and_topic():
    turn = news.build_news_turn(topic="новости рынка", max_items=4)
    assert "новости рынка" in turn and news.NEWS_PREFIX in turn and news.NEWS_NONE in turn and "до 4" in turn


# ---------- факты текста (числа, проценты, даты) ----------


def test_extract_facts_keeps_numbers_percents_dates_and_drops_noise():
    text = ("Цена выросла на 18,5% и составила 12 400 000 ₽ за 76 м²."
            " 18.10.2026 — новый прогноз, 5 новостей, в 1,5 раза чаще.")
    facts = news.extract_facts(text)
    assert "18,5%" in facts and "12 400 000" in facts and "76" in facts
    assert "18.10.2026" in facts
    assert "5" not in facts, "одиночная цифра в прозе — не факт"
    assert "1,5" not in facts, "дробь из одной цифры до и после запятой — не факт"
    assert facts.count("76") == 1, "повторов в списке фактов нет"


def test_extract_facts_is_empty_without_digits():
    assert news.extract_facts("Рынок недвижимости обсуждают эксперты и аналитики.") == []


def test_facts_kept_reports_only_missing_facts():
    original = "Ставка 18,5%, цена 12 400 000 ₽, дата 18.10.2026."
    assert news.facts_kept(original, original + " Добавили пояснение про спрос.") == []
    lost = news.facts_kept(original, "Ставка выросла, цены тоже выросли.")
    assert sorted(lost) == ["12 400 000", "18,5%", "18.10.2026"], "показать надо именно потерянные факты"


def test_fact_keys_fold_multipliers_and_ignore_date_year():
    """«1,25 млн» == «1 250 000», «18.10» == «18.10.2026» — реформулировка не считается потерей."""
    assert news.facts_kept("объём 1 250 000 ₽", "объём 1,25 млн ₽") == []
    assert news.facts_kept("событие 18.10.2026", "событие 18.10") == []
    assert news.facts_kept("площадь 76 м²", "площадь 76 кв. м") == []


def test_percent_is_not_confused_with_plain_number():
    assert news.facts_kept("рост 18,5%", "рост 18,5 пункта") == ["18,5%"]
    assert news.facts_kept("подорожало на 3 000", "подорожало на 3,0 тыс.") == []


def test_rewrite_turn_demands_fact_preservation_and_carries_url():
    turn = news.build_rewrite_turn(draft_text="Текст новости.", hint="короче", url="https://example.invalid/a")
    assert "ЖЁСТКОЕ ПРАВИЛО О ФАКТАХ" in turn and "короче" in turn
    assert "https://example.invalid/a" in turn and "Текст новости." in turn


def test_preset_hints_are_ready_to_send():
    assert len(news.PRESET_HINTS) == 5
    assert news.PRESET_HINTS[0][0].startswith("📊 Подробнее")
    for label, prompt in news.PRESET_HINTS:
        assert 0 < len(label) <= 30 and 20 <= len(prompt) <= 300, f"подсказка «{label}» вне бюджета хода"
