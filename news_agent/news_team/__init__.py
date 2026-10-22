"""Команда ролей новостного контура: исследователь → автор → проверяющий факты.

**Что здесь.** Данные ролей (`roles.py`), тексты ролей (`prompts.py`), разбор вердикта
(`verdict.py`) и инструменты (`tools/`) — всё **без агентного фреймворка**: это обычные модули,
которые проверяются тестами без установки чего-либо. Подключение к модели и порядок шагов живут
отдельно (см. `graph.py` и `llm.py`), потому что у них разные причины меняться.

**Почему пакет назван по сути, а не по фреймворку.** Команда исполняется LangGraph, а не CrewAI:
роли становятся узлами графа. Имя «команда» описывает предмет (три роли новостного
контура), а не технологию, которой он исполняется, поэтому смена фреймворка имени не меняет.

**Что здесь НЕ делается.** Модуль не знает ни про конкретного провайдера модели, ни про порядок
шагов: провайдер — в `llm.py`, порядок — в графе. Так инструмент остаётся проверяемым без сети, а
порядок — проверяемым без модели.
"""
from __future__ import annotations

from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.graph import Answer, TeamDeps, build_news_graph, material_as_draft
from news_agent.news_team.roles import AUTHOR, RESEARCHER, ROLES, VERIFIER, role
from news_agent.news_team.run import Run, Step, run_once
from news_agent.news_team.verdict import VERDICT_PREFIX, VerdictError, parse_verdict, strip_verdict

__all__ = (
    "AUTHOR",
    "RESEARCHER",
    "ROLES",
    "VERDICT_PREFIX",
    "VERIFIER",
    "Answer",
    "Run",
    "Step",
    "TeamDeps",
    "TeamUnavailable",
    "VerdictError",
    "build_news_graph",
    "material_as_draft",
    "parse_verdict",
    "role",
    "run_once",
    "strip_verdict",
)
