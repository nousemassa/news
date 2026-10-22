"""Вызов модели для роли: HTTP к провайдеру, цикл инструментов, расход за вызов.

**Почему свой вызов, а не клиент фреймворка.** Фреймворк команды собирает агента,
сам зовёт модель, сам подключает MCP-инструменты и сам считает расход — но тянет
своё отдельное окружение, а расход отдаёт **накопленным** на объекте модели: на живом
прогоне шаги давали 208733 → 209439 → 212847, где расход шага — только приросты 706 и 3408, и
сумма по шагам (631019) была бы неправдой в таблице замера. Здесь расход берётся из `usage`
каждого ответа, то есть за конкретный вызов.

**Рассуждения выключены, и это лечение дефекта.** Модель рассуждающая: при включённых рассуждениях
весь бюджет ответа уходит в `reasoning_tokens`, `content` приходит пустым, и шаг падает. Проверено
прямыми вызовами провайдера: с `reasoning.enabled=false` ответ содержательный и в разы короче.
Параметр идёт в тело запроса как есть — у прямого HTTP нет посредника, который незнакомых ключей не
принимает (путь через SDK требовал `extra_body`).

**Пустой ответ — отказ шага, а не пустая заметка.** Роль, вернувшая пустой `content` без вызова
инструментов, считается несостоявшейся: вызывающий уходит на одиночный путь, и клиент не получает
ни молчания, ни технической ошибки.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.toolset import Tool, ToolTrace

log = logging.getLogger(__name__)

#: Сколько ждём один шаг роли. Щедро: роли работают последовательно, и у новостного процесса
#: свой таймаут (шаг одиночного пути ждёт до 240 с).
STEP_TIMEOUT_S = 240.0

#: Температура роли. Как у одиночного пути: заметка переписывается, а не сочиняется заново.
TEMPERATURE = 0.2

#: Сколько кругов «модель попросила инструмент → получила результат» допускаем за шаг.
#:
#: **Почему шесть, а не четыре.** «Больше четырёх — это зацикливание» опирается на то, что
#: исследователь крутится на пустом поиске. Причина зацикливания устранена (`toolset` больше не отдаёт
#: модели фильтры, которые наш поиск исполнить не может, а промпт велит не повторять пустой запрос), но
#: честный ход состоит из большего числа шагов: до двух пустых попыток + уточняющий запрос +
#: проверка фактов. Четыре круга нарезали бы его до срока — и дефект вернулся бы под другим
#: именем. Тревога остаётся громкой (см. отказ ниже).
MAX_TOOL_ROUNDS = 6


@dataclass(frozen=True)
class Usage:
    """Расход конкретного шага: токены входа и выхода, посчитанные по ответам провайдера."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Reply:
    """Ответ роли: текст, расход шага и цена (если цены заданы в настройках)."""

    text: str
    usage: Usage
    cost_usd: float | None
    tool_calls: int = 0


def price_usd(*, usage: Usage, settings) -> float | None:
    """Цена шага из токенов и **наших** цен.

    Фреймворк денег не знает, а поставить ноль значило бы показать в таблице замера бесплатную
    команду. Поэтому при незаданных ценах ответ честно `None`, а не ноль.
    """
    price_in = float(getattr(settings, "team_price_in_per_1k_usd", 0.0) or 0.0)
    price_out = float(getattr(settings, "team_price_out_per_1k_usd", 0.0) or 0.0)
    if not price_in and not price_out:
        return None
    return (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1000.0


def request_body(*, settings, messages: list[dict], tools: list[dict]) -> dict:
    """Тело запроса к провайдеру. Отдельной функцией — чтобы форма запроса была проверяема тестом."""
    _, _, model = settings.chat_credentials()
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        # РАССУЖДЕНИЯ ВЫКЛЮЧЕНЫ ЖЁСТКО, а не «по настройке»: у рассуждающей модели весь бюджет
        # ответа уходит в `reasoning_tokens`, `content` приходит пустым, и шаг падает. Настройки на
        # это влиять не должны: цена ошибки — не «менее удачный ответ», а сорванный шаг роли.
        "reasoning": {"enabled": False},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def tool_specs(tools: dict[str, Tool]) -> list[dict]:
    """Описание инструментов для API — только имя, описание и схема аргументов."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools.values()
    ]


async def ask_role(
    *,
    settings,
    prompt: str,
    tools: dict[str, Tool] | None = None,
    client: httpx.AsyncClient | None = None,
) -> Reply:
    """Один шаг роли: отдать задание, дать модели звать инструменты, вернуть текст и расход.

    `client` — шов для тестов: без него создаётся свой `httpx.AsyncClient`. Двойник транспорта
    позволяет проверить форму запроса, цикл инструментов и разбор расхода без сети.
    """
    base_url, api_key, _ = settings.chat_credentials()
    if not api_key:
        raise TeamUnavailable("нет ключа провайдера для команды (chat_credentials пуст)")

    available = dict(tools or {})
    messages: list[dict] = [{"role": "user", "content": prompt}]
    usage = Usage()
    trace = ToolTrace()
    calls_made = 0
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=STEP_TIMEOUT_S)
    try:
        for _ in range(MAX_TOOL_ROUNDS + 1):
            data = await _post(http, base_url=base_url, api_key=api_key,
                               body=request_body(settings=settings, messages=messages,
                                                 tools=tool_specs(available)))
            usage = usage + _usage_from(data)
            message = _message_from(data)
            requested = message.get("tool_calls") or []
            if not requested:
                text = str(message.get("content") or "").strip()
                if not text:
                    raise TeamUnavailable("роль вернула пустой ответ")
                return Reply(text=text, usage=usage, cost_usd=price_usd(usage=usage, settings=settings),
                             tool_calls=calls_made)
            messages.append(message)
            for call in requested:
                calls_made += 1
                messages.append(await _tool_answer(call, available=available, trace=trace))
        raise TeamUnavailable(
            f"шаг не завершился за {MAX_TOOL_ROUNDS} кругов инструментов "
            f"(потрачено {usage.total} токенов)",
            tokens=usage.total, cost_usd=price_usd(usage=usage, settings=settings))
    finally:
        if own_client:
            await http.aclose()


async def _post(client: httpx.AsyncClient, *, base_url: str, api_key: str, body: dict) -> dict:
    """Один запрос к провайдеру. Любая неисправность — `TeamUnavailable`, а не голое исключение."""
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
    except httpx.HTTPError as error:
        raise TeamUnavailable(f"провайдер недоступен: {type(error).__name__}: {error}") from error
    if response.status_code != 200:
        raise TeamUnavailable(f"провайдер ответил {response.status_code}")
    try:
        return response.json()
    except ValueError as error:
        raise TeamUnavailable(f"ответ провайдера не разобран: {error}") from error


def _message_from(data: dict) -> dict:
    """Сообщение модели из ответа. Отсутствие выбора — отказ шага, а не пустая заметка."""
    choices = data.get("choices") or []
    if not choices:
        raise TeamUnavailable("провайдер вернул ответ без выбора")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TeamUnavailable("провайдер вернул ответ без сообщения")
    return message


def _usage_from(data: dict) -> Usage:
    """Расход ИМЕННО ЭТОГО ответа (не накопленный: у прямого HTTP копить негде)."""
    usage = data.get("usage") or {}
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


async def _tool_answer(call: dict, *, available: dict[str, Tool], trace: ToolTrace) -> dict:
    """Ответ инструмента в формате диалога. Неизвестное имя — текст, а не исключение.

    Модель может попросить инструмент, которого у роли нет (например, спутала имена): это её
    ошибка, и роль обязана узнать о ней текстом, а не уронить прогон.
    """
    function = (call.get("function") or {}) if isinstance(call, dict) else {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments") or "{}"
    tool = available.get(name)
    if tool is None:
        answer = f"инструмента {name} у роли нет"
    else:
        answer = await tool.call(_args(raw_args))
    trace.add(tool or Tool(name=name, description="", parameters={}, call=_never), answer)
    if trace.calls and trace.calls[-1][0] == name and trace.calls[-1][2] == 0:
        log.warning("команда: инструмент %s вернул пустой ответ", name)
    return {"role": "tool", "tool_call_id": str(call.get("id") or name), "content": answer}


def _args(raw: str) -> dict:
    """Аргументы вызова: строка JSON от модели. Битая строка — «аргументов нет»."""
    import json

    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _never(_: dict) -> str:  # pragma: no cover — заглушка для следа неизвестного инструмента
    return ""
