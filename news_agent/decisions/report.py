"""Замер лестницы решений на размеченном наборе.

Документы пункта требуют **два числа**: долю случаев, закрытых **без обращения к облаку** (экономия),
и **точность** на размеченном наборе. Здесь печатается и третье, без которого первые два обманывают:
**точность на закрытом локально подмножестве**. «Закрыли локально 80 %» само по себе — не результат,
а процент смелости: важно, насколько решения, принятые без облака, оказались верными.

Запуск:

    env PYTHONPATH=. .venv/bin/python -m news_agent.decisions.report \\
        --labeled eval/decisions/labeled.yaml

**Без разметки отчёт отказывается считать числа** — так же, как отказывается считать стоимость
  прогон без заданных цен. Пустая колонка метки означает «владелец ещё не размечал», и подставлять туда что-либо
(например, согласие с облаком) значило бы выдать одно число за другое.

Внешняя сеть не нужна: решатель локальный, и это же делает замер бесплатным — можно пересчитывать
сколько угодно раз, в том числе при подборе порога.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter

from agent_core.decisions.client import make_ask
from agent_core.decisions.ladder import LEVEL_CLOUD, LEVEL_SOLVER, threshold_from

# Метки и провенанс берём из одного места (`news_agent.decisions.labels`): инструкция для владельца,
# скрипт сбора и замер обязаны говорить одно и то же.
from news_agent.decisions.labels import (
    LABEL_HIDE,
    LABEL_SHOW,
    LABELS,
    ORIGIN_FINDINGS,
    ORIGINS,
    expected_answer,
    normalise_label,
)
from news_agent.news_team.tools import news_gate


def load_items(path: str) -> list[dict]:
    """Прочитать набор. YAML — как у остальных наборов проекта (`eval/retrieval_cases.yaml`)."""
    import yaml

    document = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    items = document.get("items") or []
    return [item for item in items if isinstance(item, dict)]


def check_origins(items: list[dict]) -> list[int]:
    """Номера примеров с НЕИЗВЕСТНЫМ происхождением.

    Блок влияет на то, как читается число: `findings` продаётся как «распределение прода». Пример без
    блока (или с опечаткой в нём) молча попадал в находки и раздувал прод-выборку — поэтому здесь отказ,
    а не подстановка значения по умолчанию.
    """
    from news_agent.decisions.labels import ORIGINS

    bad = []
    for index, item in enumerate(items, start=1):
        origin = str(item.get("origin") or "").strip()
        if origin and origin not in ORIGINS:
            bad.append(index)
    return bad


def labelled(items: list[dict]) -> tuple[list[dict], int, list[int], list[int]]:
    """Разделить набор: (размеченные, сколько не размечено, номера с НЕПОНЯТНОЙ меткой).

    **Почему «не размечено» и «непонятно» разделены.** Это разные состояния: первое значит «владелец
    ещё не дошёл», второе — ошибка в разметке. Свалить их в одно нельзя: опечатка молча уменьшила бы
    выборку, и число выглядело бы посчитанным по всем примерам, хотя посчитано по части.
    """
    ready, unlabeled, invalid, empty_text = [], 0, [], []
    for index, item in enumerate(items, start=1):
        label = normalise_label(item.get("label"))
        has_text = bool(str(item.get("text") or "").strip())
        if label and label not in LABELS:
            invalid.append(index)
        elif label and not has_text:
            # Помечено, но проверять нечего: это ошибка данных, а не «ещё не размечено». Свалить их в
            # одно значило бы молча уменьшить знаменатель — то, что запрещено докстрокой `labelled`.
            empty_text.append(index)
        elif label:
            ready.append({**item, "label": label})
        else:
            unlabeled += 1
    return ready, unlabeled, invalid, empty_text


async def measure(items: list[dict], *, threshold: float, ask) -> dict:
    """Прогнать решателя по размеченным примерам и посчитать числа."""
    rows = []
    for item in items:
        step, decision = await news_gate.judge(str(item["text"]), ask=ask, threshold=threshold)
        rows.append({
            "id": item.get("id"),
            # Блок набора: находки команды, отклоняемые (собраны специально) или ручные. Без этого
            # одно среднее скрывает, что решатель хорош на лёгком и слаб на рекламе.
            "origin": item.get("origin") or ORIGIN_FINDINGS,
            "label": item["label"],
            "expected": expected_answer(item["label"]),
            # Ответ ДЕЙСТВИЯ (пусто, если случай ушёл в облако) и ответ МОДЕЛИ (всегда, если он был).
            # Разделение принципиально: «ушёл в облако» — это решение, а не ошибка ответа.
            "answer": step.answer,
            "model_answer": decision.answer if decision is not None else "",
            "level": step.level,
            "confidence": round(step.confidence, 4),
            "reason": step.reason,
            "right": step.closed_locally and step.answer == expected_answer(item["label"]),
            # Ответа не было (решатель недоступен) — это НЕ ошибка модели, и в знаменатель
            # «точности модели» такие случаи не идут: иначе недоступность сервиса выглядела бы
            # ошибками решателя.
            "answered": decision is not None,
            "raw_right": (decision.answer == expected_answer(item["label"])) if decision else False,
        })

    closed = [row for row in rows if row["level"] == LEVEL_SOLVER]
    return {
        # Отклонённые показываем ПОИМЁННО: «скрыл нужное = 0» без списка ничего не доказывает —
        # константа «да» даёт тот же ноль.
        "rejects": [{"id": row["id"], "label": row["label"], "confidence": row["confidence"],
                     "origin": row["origin"]}
                    for row in closed if row["answer"] == "нет"],
        "false_rejects": sum(1 for row in closed if row["label"] == LABEL_SHOW and row["answer"] == "нет"),
        "false_accepts": sum(1 for row in closed if row["label"] == LABEL_HIDE and row["answer"] == "да"),
        "by_origin": by_origin(rows),
        "items": len(rows),
        "closed": len(closed),
        "closed_share": (len(closed) / len(rows)) if rows else 0.0,
        "accuracy_on_closed": (sum(1 for row in closed if row["right"]) / len(closed)) if closed else 0.0,
        "answered": sum(1 for row in rows if row["answered"]),
        "unanswered": sum(1 for row in rows if not row["answered"]),
        "accuracy_raw": (
            sum(1 for row in rows if row["raw_right"]) / sum(1 for row in rows if row["answered"])
        ) if any(row["answered"] for row in rows) else 0.0,
        "escalated_reasons": dict(Counter(row["reason"] for row in rows if row["level"] == LEVEL_CLOUD)),
        "threshold": threshold,
        "rows": rows,
    }


def by_origin(rows: list[dict]) -> dict:
    """Числа по БЛОКАМ набора: находки команды, отклоняемые, ручные.

    Зачем отдельно: набор описывает два разных распределения. Блок находок — то, что команда
    реально приносит; блок отклоняемых собран специально, чтобы решателю было на чём ошибиться.
    Среднее по ним ничего не говорит о продакшене, а по отдельности видно, где решатель слаб.
    """
    blocks: dict[str, list[dict]] = {}
    for row in rows:
        blocks.setdefault(row.get("origin") or ORIGIN_FINDINGS, []).append(row)
    result = {}
    for name, block in blocks.items():
        closed = [row for row in block if row["level"] == LEVEL_SOLVER]
        result[name] = {
            "items": len(block),
            "false_rejects": sum(1 for row in closed if row["label"] == LABEL_SHOW and row["answer"] == "нет"),
            "false_accepts": sum(1 for row in closed if row["label"] == LABEL_HIDE and row["answer"] == "да"),
            "labels": {label: sum(1 for row in block if row["label"] == label) for label in LABELS},
            "closed": len(closed),
            "closed_share": (len(closed) / len(block)) if block else 0.0,
            "accuracy_on_closed": (sum(1 for row in closed if row["right"]) / len(closed)) if closed else 0.0,
            "accuracy_raw": (sum(1 for row in block if row["raw_right"]) / len(block)) if block else 0.0,
        }
    return result


def report(result: dict, *, unlabeled: int = 0, invalid: list[int] | None = None) -> None:
    """Три числа и их честная расшифровка.

    Печать различает «ещё не размечено» и «метка непонятна»: первое — состояние работы, второе —
    ошибка, из-за которой пример выпал из замера. Молчать об ошибке нельзя: иначе число выглядит
    посчитанным по всему набору, а посчитано по его части.
    """
    print("\n=== лестница решений на размеченном наборе ===")
    print(f"порог уверенности: {result['threshold']}")
    print(f"размечено примеров: {result['items']}"
          + (f" (ещё не размечено: {unlabeled})" if unlabeled else ""))
    if invalid:
        print(f"⚠️ метка непонятна у примеров: {invalid[:10]}"
              + ("…" if len(invalid) > 10 else "")
              + f" — их {len(invalid)}, и они НЕ в замере")
    if not result["items"]:
        print("размеченных примеров нет — число получить нельзя")
        return
    print()
    print(f"1. закрыто БЕЗ облака (экономия)     : {result['closed_share'] * 100:.1f}%"
          f"  ({result['closed']} из {result['items']})")
    print(f"2. точность на закрытых локально     : {result['accuracy_on_closed'] * 100:.1f}%"
          f"  ← это и есть качество экономии")
    # Знаменатель — только те случаи, где решение вообще было: недоступность сервиса не ошибка модели.
    answered = result.get("answered", result["items"])
    print(f"3. точность модели БЕЗ порога        : {result['accuracy_raw'] * 100:.1f}%"
          f"  (что дал бы решатель, доверяй мы ему целиком; отвечено {answered} из {result['items']}"
          + (f", без ответа {result['unanswered']}" if result.get("unanswered") else "") + ")")
    rejects, accepts = result.get("false_rejects", 0), result.get("false_accepts", 0)
    print("\nошибки локальных решений, по цене для продукта:")
    print(f"   скрыл то, что владелец счёл нужным : {rejects}"
          + ("  ← важное не терялось" if rejects == 0 else "  ← ДОРОГАЯ ошибка: владелец не увидел новость"))
    print(f"   показал то, что владелец отклонил : {accepts}  (мягкая: лишняя карточка, а не потеря)")
    blocks = result.get("by_origin") or {}
    if blocks:
        print("\nпо блокам набора (блоки описывают РАЗНЫЕ распределения — среднее по ним ничего не говорит):")
        print(f"  {'блок':12s} {'примеров':>9s} {'меток 1/0':>10s} {'закрыто':>9s}"
              f" {'модель':>8s} {'скрыл':>8s} {'показал':>9s}")
        for name, block in sorted(blocks.items()):
            labels = block["labels"]
            print(f"  {name:12s} {block['items']:9d} "
                  f"{str(labels.get(LABEL_SHOW, 0)) + '/' + str(labels.get(LABEL_HIDE, 0)):>10s} "
                  f"{block['closed_share'] * 100:8.1f}% {block['accuracy_raw'] * 100:9.1f}% "
                  f"{block.get('false_rejects', 0):>8d} {block.get('false_accepts', 0):>9d}")
        # Однобокий блок — тот, где нет хотя бы одного класса: точность в нём не различает, и
        # называем ИМЕННО такие блоки, а не «где-то проблема».
        one_sided = [name for name, block in blocks.items()
                     if not all(block["labels"].get(label, 0) for label in LABELS)]
        if one_sided:
            print(f"  ⚠️ однобокие блоки (нет одного из классов): {', '.join(sorted(one_sided))} — "
                  "точность в них не различает: решатель, всегда говорящий «да», выглядел бы "
                  "безошибочным")
    if result["escalated_reasons"]:
        print("\nпричины ухода на уровень 3:")
        for reason, count in sorted(result["escalated_reasons"].items(), key=lambda pair: -pair[1]):
            print(f"  {reason}: {count}")
    print("\nоговорка: точность считается на размеченных ВЛАДЕЛЬЦЕМ примерах; доля закрытых зависит")
    print("от порога, а порог подобран по этому же набору — то есть число описывает именно эту настройку.")


def sweep(*, result: list[dict], unlabeled: int) -> None:
    """Таблица порогов: что мы отдаём и что получаем на каждом шаге.

    Печатается вместе с НАБОРОМ ОТКЛОНЁННЫХ на каждом пороге: «скрыл нужное = 0» само по себе
    ничего не доказывает — константа «да» даёт тот же ноль, только с 30 лишними показами.
    """
    print()
    print("=== промер по порогам (обмен экономии на шум) ===")
    print(f"  {'порог':>6} {'закрыто':>8} {'точность':>9} {'модель':>8} {'лишних':>7}  отклонено (метка)")
    for row in result:
        rejected = row.get("rejects") or []
        shown = ", ".join(f"{item['id']}:{item['label']}" for item in rejected[:8]) or "—"
        print(f"  {row['threshold']:>6} {row['closed_share'] * 100:7.1f}% "
              f"{row['accuracy_on_closed'] * 100:8.1f}% {row['accuracy_raw'] * 100:7.1f}% "
              f"{row['false_accepts']:>7}  {shown}")
    print("  оговорка: порог подобран по этому же набору; «отклонено (метка)» — что именно решатель")
    print("  уверенно спрятал, с меткой владельца: '1' в этом списке была бы ДОРОГОЙ ошибкой.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер лестницы решений")
    parser.add_argument("--labeled", default="eval/decisions/labeled.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="по умолчанию — из настроек (SOLVER_CONFIDENCE_THRESHOLD)")
    parser.add_argument("--out", default="eval/decisions/report.json")
    # Промер по сетке порогов. Нужен, потому что выбор порога — это обмен, и он должен быть
    # воспроизводим: таблица порогов, живущая только в тексте отчёта, не перепроверяется.
    parser.add_argument("--sweep", default="",
                        help="сетка порогов через запятую: печатает и пишет таблицу (напр. 0.6,0.7,0.8,0.9)")
    args = parser.parse_args()

    from news_agent.settings import settings

    threshold = args.threshold if args.threshold is not None else threshold_from(settings)

    items = load_items(args.labeled)
    ready, unlabeled, invalid, empty_text = labelled(items)
    unknown_origin = check_origins(items)
    if unknown_origin:
        print(
            f"[report] ОТКАЗ: у {len(unknown_origin)} примеров непонятное происхождение (номера: "
            f"{unknown_origin[:10]}).\n"
            f"  Допустимые блоки: {ORIGINS}. Блок влияет на чтение числа: без него пример молча "
            "попал бы в блок находок, который читается как распределение прода.",
            file=sys.stderr,
        )
        return 4
    if empty_text:
        print(
            f"[report] ОТКАЗ: у {len(empty_text)} примеров метка стоит, а текста нет (номера: "
            f"{empty_text[:10]}).\n"
            "  Это ошибка данных, а не «ещё не размечено»: молча выбросив их, отчёт уменьшил бы "
            "знаменатель и посчитал бы точность по части набора, выглядя посчитанной по всему.",
            file=sys.stderr,
        )
        return 5
    if invalid:
        print(
            f"[report] ОТКАЗ: у {len(invalid)} примеров метка непонятна (номера: {invalid[:10]}"
            + ("…" if len(invalid) > 10 else "") + ").\n"
            f"  Допустимые метки: {LABELS[0]!r} — показывать, {LABELS[1]!r} — не показывать.\n"
            "  Считать по части набора нельзя: число выглядело бы посчитанным по всему набору.",
            file=sys.stderr,
        )
        return 3
    if not ready:
        print(
            f"[report] ОТКАЗ: в {args.labeled} нет размеченных примеров.\n"
            "  Причина не формальная: точность измеряется по меткам владельца, и подставить сюда\n"
            "  согласие с облаком значило бы выдать одно число за другое. Разметьте колонку label\n"
            f"  значениями {LABELS[0]!r} / {LABELS[1]!r}.",
            file=sys.stderr,
        )
        return 2

    ask = make_ask(settings.solver_url, api_key=settings.solver_api_key,
                                timeout_s=settings.solver_timeout_s)
    result = asyncio.run(measure(ready, threshold=threshold, ask=ask))
    report(result, unlabeled=unlabeled, invalid=invalid)

    if args.sweep:
        grid = [float(value) for value in args.sweep.split(",") if value.strip()]
        rows = []
        for point in grid:
            measured = asyncio.run(measure(ready, threshold=point, ask=ask))
            rows.append({
                "threshold": point,
                "closed_share": round(measured["closed_share"], 4),
                "accuracy_on_closed": round(measured["accuracy_on_closed"], 4),
                "accuracy_raw": round(measured["accuracy_raw"], 4),
                "false_accepts": measured.get("false_accepts", 0),
                "rejects": measured.get("rejects", []),
            })
        sweep(result=rows, unlabeled=unlabeled)
        sweep_path = pathlib.Path(args.out).with_name("report-sweep.json")
        sweep_path.write_text(json.dumps({"items": len(ready), "grid": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[sweep] машинный отчёт: {sweep_path}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] машинный отчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
