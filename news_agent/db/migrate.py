"""Накат схемы новостного продукта: общий слой, затем новостные таблицы.

Запуск: `python -m news_agent.db.migrate --components news`
"""
from __future__ import annotations

from agent_core.db.migrate import cli
from agent_core.db.registry import CORE_COMPONENTS
from news_agent.db.repo import NEWS_COMPONENTS


def main(argv: list[str] | None = None) -> int:
    return cli(CORE_COMPONENTS + NEWS_COMPONENTS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
