"""Вердикт проверяющего: разбор служебной строки `VERDICT|{json}`. Без фреймворка.

**Зачем маркер, а не просьба «верни текст».** Живой прогон показал, что роль возвращает не текст
заметки, а ОТЧЁТ о проверке: «Инструмент `lost_facts` вернул пустой список — потерь нет. Проверка
показала, что…». Просьба в задании это не лечит: послушание модели — не гарантия, а публиковать
отчёт вместо заметки нельзя. Приём для этого — служебная строка в ответе
агента (`NEWS|`): она делает выход модели **разбираемым** и
позволяет отказать ходу, если её нет, вместо надежды на формулировку.

**Почему разбор отдельным модулем.** Маркер — это контракт между заданием роли и нашим кодом, и
у него одна причина меняться. Здесь только разбор; задание роли живёт в `prompts.py`, порядок
шагов — в `run.py`.
"""
from __future__ import annotations

import json

# Маркер вердикта. Префикс отдельным именем: он же используется в задании роли, и разойтись им
# нельзя — иначе роль получит один маркер, а разберём мы другой.
VERDICT_PREFIX = "VERDICT|"


class VerdictError(ValueError):
    """Вердикт не разобран. Отдельный тип, чтобы вызывающий не ловил всё подряд."""


def parse_verdict(answer: str) -> tuple[str, tuple[str, ...], bool]:
    """Разобрать ответ проверяющего → (итоговый текст, потерянные факты, показывать ли).

    Третье поле — `show` из вердикта: гейт новости решает, показывать ли новость владельцу.
    **Отсутствие поля означает «показывать»**, и это не поблажка: показ — состояние по умолчанию,
    а скрыть новость можно только явным решением. Молчаливое «не
    показывать» от невнимательности было бы худшим исходом.

    Бросает `VerdictError`, если служебной строки нет или она битая: пустой ответ и «текст, в
    котором ничего не сказано» — разные вещи, и подменять первое вторым нельзя.
    """
    for line in (answer or "").splitlines():
        if not line.strip().startswith(VERDICT_PREFIX):
            continue
        payload = line.strip()[len(VERDICT_PREFIX):].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise VerdictError(f"битая строка вердикта: {error}") from error
        if not isinstance(data, dict):
            raise VerdictError("вердикт должен быть объектом JSON")
        text = str(data.get("text") or "").strip()
        if not text:
            raise VerdictError("в вердикте нет итогового текста")
        lost = data.get("lost") or []
        if not isinstance(lost, list):
            raise VerdictError("поле lost должно быть списком")
        show = data.get("show", True)
        if not isinstance(show, bool):
            raise VerdictError("поле show должно быть true или false")
        return text, tuple(str(item) for item in lost), show
    raise VerdictError("проверяющий не вернул строку VERDICT| — вердикта нет")


def strip_verdict(answer: str) -> str:
    """Убрать служебную строку из произвольного текста — на случай, если она утечёт наружу."""
    return "\n".join(
        line for line in (answer or "").splitlines()
        if not line.strip().startswith(VERDICT_PREFIX)
    )
