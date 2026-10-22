"""Контракт новостного агента и разбор его ответа.

Приём тот же, что во всех контурах: формат задаётся ТЕКСТОМ хода, а код разбирает строки и
валидирует. Для новостей — построчный JSON, по одной строке на новость:

    NEWS|{"title": "…", "text": "…", "url": "https://…", "source": "…"}

Если агент ничего не нашёл, он отвечает ровно `NEWS_NONE`. Проза без строк наружу не выходит:
владельцу показывается честное сообщение и кнопка «Ещё раз» (см. `tg/news_bot.py`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

NEWS_PREFIX = "NEWS|"
NEWS_NONE = "NEWS_NONE"
#: Маркер «внешний поиск НЕ СРАБОТАЛ» — отдельно от «ничего не нашёл».
#:
#: Смешав два разных состояния в одном `NEWS_NONE` — и «поиск отработал, новостей
#: нет», и «поиск отказал» (лимит запросов, таймаут, недоступный MCP-сервер), — получаем отказ
#: поиска, который выглядит в отчёте как отсутствие материала: причина исчезает, а диагностика, которая
#: называет не ту причину, хуже отсутствия диагностики.
SEARCH_FAILED = "SEARCH_FAILED"
CARD_LIMIT = 3500          # грубый потолок карточки (точный лимит Telegram считаем при отправке)
MIN_TEXT = 40
TELEGRAM_LIMIT = 4096      # знаков в сообщении Telegram (считается по UTF-16, см. `utf16_len`)
URL_RE = re.compile(r"^https?://\S+$")


@dataclass(frozen=True)
class NewsItem:
    title: str
    text: str
    url: str = ""
    source: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "text": self.text, "url": self.url, "source": self.source}


def build_news_turn(*, topic: str, max_items: int) -> str:
    """Текст хода для агента: тема + строгий формат ответа."""
    return (
        f"Найди актуальные новости по теме: {topic}.\n"
        f"Верни от 1 до {max_items} новостей. На КАЖДУЮ новость — отдельная строка ровно такого вида:\n"
        'NEWS|{"title":"краткий заголовок","text":"2–4 предложения по существу","url":"https://ссылка-на-источник",'
        '"source":"имя источника"}\n'
        "Вне строк NEWS| никакого текста не добавляй: ни вступлений, ни выводов, ни нумерации.\n"
        "Только факты из найденных материалов, ссылка обязательна. Ничего не выдумывай.\n"
        f"Если ничего не нашёл или поиск недоступен — ответь ровно {NEWS_NONE}."
    )


def build_rewrite_turn(*, draft_text: str, hint: str, url: str = "") -> str:
    """Текст хода для «Переписать по подсказке»: правим формулировку, факты сохраняем.

    Ссылка передаётся отдельно: она не всегда есть в тексте карточки, а подсказка «С источником»
    без неё бессмысленна (`publish_text` добавляет её при публикации, если агент не оставил).
    """
    parts = [
        "Перепиши текст новости по подсказке владельца.",
        ("ЖЁСТКОЕ ПРАВИЛО О ФАКТАХ: числа, проценты, даты, суммы, площади, адреса и названия из исходного "
         "текста нельзя удалять, округлять или заменять — их можно только дополнять пояснениями. Если "
         "просят короче — сокращай формулировки, но все числа, названия и адреса оставляй."),
        "Новых фактов и цифр не добавляй.",
        f"Подсказка: {hint.strip()}",
    ]
    if url:
        parts.append(f"Ссылка на источник (сохрани её в тексте): {url}")
    parts.append(f"Текст новости:\n{draft_text.strip()}")
    parts.append("Верни ТОЛЬКО новый текст новости, без пояснений, без строк NEWS| и без кавычек.")
    return "\n".join(parts)


# ---------- кнопки-подсказки и защита фактов ----------

# Кнопка → готовый промпт для агента (владелец жмёт вместо набора текста). Правки — здесь, в одном месте.
PRESET_HINTS: tuple[tuple[str, str], ...] = (
    ("📊 Подробнее",
     ("Раскрой подробнее: добавь контекст, пояснения и последствия. Сохрани ВСЕ факты, числа, проценты, "
      "даты и адреса из исходного текста — ничего не удаляй, не округляй и не заменяй.")),
    ("✂️ Короче",
     "Сократи до 2–3 предложений. Все числа, проценты, даты, названия и адреса сохрани без изменений."),
    ("🗣 Проще",
     "Убери канцелярит и длинные обороты, пиши простыми фразами. Все факты и цифры оставь как есть."),
    ("📈 С акцентом на цифры",
     "Начни с ключевой цифры или динамики; все остальные факты сохрани полностью."),
    ("📎 С источником",
     "Закончи текст ссылкой на источник — она указана ниже."),
)

_FACT_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
# Число (в т.ч. «2 203» с разделителем разрядов) + необязательная дробная часть и единица.
# Из единиц значимы `%` (проценты не сравниваем с числами) и множители (тыс/млн/млрд/трлн).
_FACT_NUM_RE = re.compile(
    r"(?<![\d,.])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,](\d+))?"
    r"\s*(%|тыс\w*|млн\w*|млрд\w*|трлн\w*)?")
_FACT_MULTIPLIERS = (("трлн", 1e12), ("млрд", 1e9), ("млн", 1e6), ("тыс", 1e3))


def _fact_key(value: float) -> str:
    """Ключ числа без экспоненты и хвостовых нулей: «1,25 млн» и «1 250 000» дают один ключ."""
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _facts(text: str) -> list[tuple[str, str]]:
    """Факты текста: (ключ для сравнения, как показать человеку). Порядок — как в тексте, без повторов.

    Факты — числа/проценты (с множителями) и даты. Проценты сравниваются отдельно от чисел
    («18%» ≠ «18»), множители приводятся к базовому значению. Ссылки сюда не входят: их при
    публикации восстанавливает `publish_text`, поэтому их отсутствие в тексте агента — не потеря.
    """
    body = text or ""
    found: list[tuple[str, str]] = []
    date_spans: list[tuple[int, int]] = []
    for match in _FACT_DATE_RE.finditer(body):
        day, month = int(match.group(1)), int(match.group(2))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        found.append((f"дата {day:02d}.{month:02d}", match.group(0)))
        date_spans.append(match.span())
    for match in _FACT_NUM_RE.finditer(body):
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        digits = re.sub(r"\D", "", match.group(1))
        unit = (match.group(3) or "").lower()
        if not unit and len(digits) < 2:      # одиночная цифра в прозе («5 новостей») — не факт
            continue
        value = float(f"{digits}.{match.group(2)}") if match.group(2) else float(digits)
        mult = next((factor for prefix, factor in _FACT_MULTIPLIERS if unit.startswith(prefix)), 1.0)
        key = _fact_key(value * mult)
        found.append((f"{key}%" if unit == "%" else key, match.group(0).strip()))
    unique: dict[str, str] = {}
    for key, display in found:
        unique.setdefault(key, display)
    return list(unique.items())


def extract_facts(text: str) -> list[str]:
    """Факты текста как их показывают человеку (числа, проценты, даты) — для проверок и сообщений."""
    return [display for _, display in _facts(text)]


def facts_kept(original: str, new: str) -> list[str]:
    """Какие факты ИСХОДНОГО текста НЕ попали в новый: пустой список ⇒ всё сохранено."""
    keys = {key for key, _ in _facts(new)}
    return [display for key, display in _facts(original) if key not in keys]


def parse_news_blocks(text: str, *, max_items: int, card_limit: int = CARD_LIMIT) -> tuple[list[NewsItem], str]:
    """Разобрать ответ агента → (новости, статус).

    **Статус называет ПРИЧИНУ, а не «получилось/нет»:** `ok` | `none` | `empty` | `format` | `search_failed`.

    Зачем так. Словарь (`ok`/`none`/`empty`) склеивал ТРИ разных состояния в одно: «поиск
    сработал, новостей нет» (`NEWS_NONE`), «агент ответил прозой вместо строк `NEWS|`» и «отказ
    ИСТОЧНИКА» (`SEARCH_FAILED`). Наружу из них уходило одно «агент не отдал карточек» — то есть
    сообщение называло не ту причину, а разбор шёл вслепую. Два последних — не «новостей нет»:
    отказ источника надо лечить иначе, чем пустую выдачу, а проза означает нарушенный контракт роли.

    Битая строка не роняет разбор: считаем её в лог и идём дальше. Дубли по ссылке отбрасываем.
    """
    raw = (text or "").strip()
    if not raw:
        return [], "empty"
    # Отказ источника проверяем ПЕРВЫМ: маркер отказа — не «новостей нет» и не «плохой формат».
    if SEARCH_FAILED in raw:
        return [], "search_failed"
    if NEWS_NONE in raw:
        return [], "none"
    items: list[NewsItem] = []
    seen: set[str] = set()
    broken = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith(NEWS_PREFIX):
            continue
        payload = line[len(NEWS_PREFIX):].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            broken += 1
            continue
        if not isinstance(data, dict):
            broken += 1
            continue
        title = str(data.get("title") or "").strip()
        body = str(data.get("text") or "").strip()
        url = str(data.get("url") or "").strip()
        source = str(data.get("source") or "").strip()
        if not title or len(body) < MIN_TEXT:
            broken += 1
            continue
        if url and not URL_RE.match(url):
            broken += 1
            continue
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        if len(title) + len(body) > card_limit:
            keep = max(MIN_TEXT, card_limit - len(title) - 1)
            body = body[:keep].rstrip() + "…"
        items.append(NewsItem(title=title, text=body, url=url, source=source))
        if len(items) >= max_items:
            break
    if broken:
        log.info("news: строк с нарушением формата — %d", broken)
    if items:
        return items, "ok"
    # Строк `NEWS|` не нашлось вовсе: это НЕ «новостей нет» (то состояние называется `NEWS_NONE`),
    # а ответ НЕ ТЕМ форматом — и владельцу его надо назвать иначе.
    return [], "format"


def strip_news_blocks(text: str) -> str:
    """Убрать служебные строки из произвольного текста (страховка перед отправкой человеку)."""
    kept = [line for line in (text or "").splitlines() if not line.strip().startswith(NEWS_PREFIX)]
    return "\n".join(kept).strip()


def neutralize_news_blocks(text: str) -> str:
    """Ввод владельца: `NEWS|` в его сообщении не должен превращаться в наши строки."""
    return (text or "").replace(NEWS_PREFIX, "NEWS·|")


def format_card(item: NewsItem, *, tz: ZoneInfo, found_at: datetime | None = None) -> str:
    """Текст карточки для бота: заголовок, текст, метка свежести, источник."""
    head = f"📰 {item.title}"
    body = f"{head}\n\n{item.text}"
    if found_at is not None:
        body += f"\n\n🕒 найдено {found_at.astimezone(tz):%d.%m %H:%M}"
    if item.url:
        body += f"\n🔗 {item.url}"
    return body


def publish_text(item: NewsItem) -> str:
    """Текст для канала: текст новости; источник добавляем, если его нет в тексте."""
    body = item.text.strip()
    if item.url and item.url not in body:
        body += f"\n\nИсточник: {item.url}"
    return body


def utf16_len(text: str) -> int:
    """Длина как её считает Telegram: эмодзи и astral-символы занимают две единицы."""
    return len(text.encode("utf-16-le")) // 2


def chunk_for_telegram(text: str, *, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Разбить длинный текст на сообщения, не разрывая абзацы (и не превышая лимит Telegram)."""
    clean = (text or "").strip()
    if not clean:
        return []
    if utf16_len(clean) <= limit:
        return [clean]
    chunks: list[str] = []
    current = ""
    for block in clean.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if utf16_len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if utf16_len(block) <= limit:
            current = block
            continue
        # блок сам больше лимита — режем по строкам, затем жёстко по символам
        for line in block.splitlines():
            piece = f"{current}\n{line}" if current else line
            if utf16_len(piece) <= limit:
                current = piece
                continue
            if current:
                chunks.append(current)
                current = ""
            while line:
                cut = limit
                while cut > 1 and utf16_len(line[:cut]) > limit:
                    cut = int(cut * 0.9)
                chunks.append(line[:cut])
                line = line[cut:]
    if current:
        chunks.append(current)
    return chunks
