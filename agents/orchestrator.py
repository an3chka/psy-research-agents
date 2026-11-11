"""Оркестратор: запускает агентов по очереди (поисковые — параллельно) и записывает ход работы."""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import analyst, planner, pubmed_agent, stats_agent, trials_agent


@dataclass
class Settings:
    max_articles: int = 15
    trials_sample: int = 200
    trials_for_summary: int = 15
    year_from: int = 2015
    year_to: int = 2026
    only_active_trials: bool = False
    with_trend: bool = True


@dataclass
class TraceStep:
    agent: str
    message: str
    seconds: float


@dataclass
class RunResult:
    query: str
    plan: planner.SearchPlan
    pubmed: pubmed_agent.PubMedResult | None
    trials: trials_agent.TrialsResult | None
    stats: stats_agent.StatsResult
    summary: analyst.Summary
    trace: list[TraceStep] = field(default_factory=list)
    errors: dict = field(default_factory=dict)


def run(query: str, settings: Settings, on_step=lambda step: None) -> RunResult:
    trace: list[TraceStep] = []
    t0 = time.monotonic()

    def log(agent: str):
        def _log(msg: str):
            step = TraceStep(agent, msg, round(time.monotonic() - t0, 1))
            trace.append(step)
            on_step(step)
        return _log

    # 1. Планировщик
    plan_log = log("🧭 Планировщик")
    plan_log(f"получил запрос: «{query}»")
    plan = planner.plan(query)
    plan_log(f"PubMed: {plan.pubmed_query}")
    plan_log(f"ClinicalTrials: condition=«{plan.trials_condition or '—'}», term=«{plan.trials_term or '—'}»"
             + (f" ({plan.note})" if plan.note else ""))

    # 2. Поисковые агенты работают параллельно
    errors = {}
    pubmed = trials = None
    if plan.source == "raw" and re.search(r"[а-яё]", query, flags=re.I):
        # без нейросети русский текст не переведётся, а PubMed по кириллице вернёт мусор
        errors["Планировщик"] = ("не удалось перевести запрос на английский (нейросеть недоступна). "
                                 "Выберите готовую тему или напишите запрос по-английски.")
        plan_log("поиск остановлен: запрос на русском, а перевести его нечем")
    else:
        pm_log, ct_log = log("📚 Агент PubMed"), log("🧪 Агент ClinicalTrials")
        with ThreadPoolExecutor(max_workers=2) as pool:
            pm_future = pool.submit(
                pubmed_agent.search, plan.pubmed_query, plan.keywords_en, settings.max_articles,
                settings.year_from, settings.year_to, settings.with_trend, pm_log,
            )
            ct_future = pool.submit(
                trials_agent.search, plan.trials_condition, plan.trials_term, settings.only_active_trials,
                settings.trials_sample, settings.year_from, ct_log,
            )
            try:
                pubmed = pm_future.result()
                pm_log("готово")
            except Exception as exc:  # noqa: BLE001
                pubmed, errors["PubMed"] = None, str(exc)
                pm_log(f"ошибка: {exc}")
            try:
                trials = ct_future.result()
                ct_log("готово")
            except Exception as exc:  # noqa: BLE001
                trials, errors["ClinicalTrials.gov"] = None, str(exc)
                ct_log(f"ошибка: {exc}")

    # 3. Статистик
    st_log = log("📊 Статистик")
    stats = stats_agent.analyze(pubmed, trials)
    st_log(f"посчитал показатели: {', '.join(stats.facts.keys()) or 'нет данных'}")

    # 4. Аналитик
    an_log = log("🧠 Аналитик")
    an_log("пишу сводку…")
    summary = analyst.summarize(plan, pubmed, trials, stats, settings.trials_for_summary)
    an_log(f"готово ({summary.model}); ссылок в тексте: {len(summary.cited)}, "
           f"не найдено в данных: {len(summary.unknown_ids)}")

    return RunResult(query, plan, pubmed, trials, stats, summary, trace, errors)
