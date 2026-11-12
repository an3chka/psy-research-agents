"""Оркестратор: управляет агентами, передаёт данные между ними и записывает ход работы.

Порядок работы:
1. Планировщик превращает вопрос в поисковые запросы.
2. Агенты PubMed и ClinicalTrials ищут параллельно.
3. Критик оценивает, отвечают ли результаты на вопрос. Если нет — планировщик
   переписывает запросы по его замечанию, и поиск повторяется (до max_revisions раз).
4. Агент PubMed выделяет в статьях методы лечения и передаёт их агенту ClinicalTrials,
   который проверяет, сколько испытаний идёт по каждому методу.
5. Статистик считает показатели, аналитик пишет сводку.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import analyst, critic, llm, methods, planner, pubmed_agent, stats_agent, trials_agent

PLANNER, PUBMED, TRIALS, CRITIC, STATS, ANALYST = (
    "Планировщик", "Агент PubMed", "Агент ClinicalTrials", "Критик", "Статистик", "Аналитик")


@dataclass
class Settings:
    max_articles: int = 15
    trials_sample: int = 200
    trials_for_summary: int = 15
    year_from: int = 2015
    year_to: int = 2026
    only_active_trials: bool = False
    with_trend: bool = True
    use_llm: bool = True  # False, если на сайте не введён пароль
    max_revisions: int = 2  # сколько раз критик может вернуть запрос на доработку


@dataclass
class TraceStep:
    agent: str
    message: str
    seconds: float


@dataclass
class Round:
    """Одна попытка поиска и оценка критика."""
    attempt: int
    plan: planner.SearchPlan
    pubmed_total: int | None
    trials_total: int | None
    review: critic.Review


@dataclass
class RunResult:
    query: str
    plan: planner.SearchPlan
    pubmed: pubmed_agent.PubMedResult | None
    trials: trials_agent.TrialsResult | None
    stats: stats_agent.StatsResult
    summary: analyst.Summary
    methods: methods.MethodsResult | None = None
    rounds: list[Round] = field(default_factory=list)
    chosen_round: int = 1
    trace: list[TraceStep] = field(default_factory=list)
    errors: dict = field(default_factory=dict)


def _search(plan: planner.SearchPlan, settings: Settings, pm_log, ct_log):
    """Агенты PubMed и ClinicalTrials работают параллельно."""
    errors = {}
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
        except Exception as exc:  # noqa: BLE001
            pubmed, errors["PubMed"] = None, str(exc)
            pm_log(f"ошибка: {exc}")
        try:
            trials = ct_future.result()
        except Exception as exc:  # noqa: BLE001
            trials, errors["ClinicalTrials.gov"] = None, str(exc)
            ct_log(f"ошибка: {exc}")
    return pubmed, trials, errors


def run(query: str, settings: Settings, on_step=lambda step: None) -> RunResult:
    trace: list[TraceStep] = []
    t0 = time.monotonic()

    def log(agent: str):
        def _log(msg: str):
            step = TraceStep(agent, msg, round(time.monotonic() - t0, 1))
            trace.append(step)
            on_step(step)
        return _log

    plan_log, pm_log, ct_log, cr_log = log(PLANNER), log(PUBMED), log(TRIALS), log(CRITIC)
    can_revise = settings.use_llm and llm.is_configured()

    # 1. Планировщик
    plan_log(f"получил запрос: «{query}»")
    plan = planner.plan(query, settings.use_llm)
    plan_log(f"PubMed: {plan.pubmed_query}")
    plan_log(f"ClinicalTrials: condition=«{plan.trials_condition or '—'}», term=«{plan.trials_term or '—'}»"
             + (f" ({plan.note})" if plan.note else ""))

    pubmed = trials = None
    found_methods = None
    errors: dict = {}
    rounds: list[Round] = []
    chosen = 1

    if plan.source == "raw" and re.search(r"[а-яё]", query, flags=re.I):
        # без нейросети русский текст не переведётся, а PubMed по кириллице вернёт мусор
        errors["Планировщик"] = ("не удалось перевести запрос на английский (нейросеть недоступна или не введён пароль). "
                                 "Выберите готовую тему или напишите запрос по-английски.")
        plan_log("поиск остановлен: запрос на русском, а перевести его нечем")
    else:
        best = None  # (оценка, номер попытки, план, pubmed, trials, ошибки, методы)
        for attempt in range(1, settings.max_revisions + 2):
            # 2. Поиск
            pubmed_, trials_, errors_ = _search(plan, settings, pm_log, ct_log)

            # 3. Критик проверяет результаты; параллельно агент PubMed выделяет методы лечения
            cr_log(f"попытка {attempt}: проверяю, отвечают ли результаты на вопрос")
            with ThreadPoolExecutor(max_workers=2) as pool:
                review_f = pool.submit(critic.review, query, plan, pubmed_, trials_, settings.use_llm)
                methods_f = pool.submit(methods.extract, pubmed_.articles if pubmed_ else [], settings.use_llm)
                review, methods_ = review_f.result(), methods_f.result()
            if review.score is None:
                cr_log(f"оценка не проводилась ({review.model})")
            else:
                cr_log(f"оценка {review.score}/10" + (f": {review.problem}" if review.problem else ""))
            rounds.append(Round(attempt, plan, pubmed_.total_count if pubmed_ else None,
                                trials_.total_count if trials_ else None, review))

            score = 10 if review.score is None else review.score
            if best is None or score > best[0]:
                best = (score, attempt, plan, pubmed_, trials_, errors_, methods_)
            if review.ok:
                cr_log("результаты подходят, передаю дальше")
                break
            if not can_revise or attempt > settings.max_revisions:
                cr_log(f"больше попыток нет, оставляю лучшую (попытка {best[1]}, оценка {best[0]}/10)")
                break

            # обратная связь: критик → планировщик
            cr_log(f"возвращаю планировщику с советом: {review.advice or 'уточнить запросы'}")
            new_plan = planner.replan(query, plan, review.problem, review.advice)
            if new_plan is None:
                plan_log("не смог переписать запрос (нейросеть не ответила), оставляю лучший вариант")
                break
            plan = new_plan
            plan_log(f"попытка {attempt + 1}. PubMed: {plan.pubmed_query}")
            plan_log(f"попытка {attempt + 1}. ClinicalTrials: condition=«{plan.trials_condition or '—'}», "
                     f"term=«{plan.trials_term or '—'}»")

        _, chosen, plan, pubmed, trials, errors, found_methods = best

        # 4. Передача между агентами: методы из статей → проверка в испытаниях
        if found_methods and found_methods.methods:
            pm_log(f"нашёл в статьях методы ({found_methods.source}), передаю агенту ClinicalTrials: "
                   + ", ".join(m.name_ru for m in found_methods.methods))
            methods.check_trials(found_methods, plan.trials_condition or plan.keywords_en,
                                 settings.year_from, ct_log)

    # 5. Статистик
    st_log = log(STATS)
    stats = stats_agent.analyze(pubmed, trials)
    st_log(f"посчитал показатели: {', '.join(stats.facts.keys()) or 'нет данных'}")

    # 6. Аналитик
    an_log = log(ANALYST)
    an_log("пишу сводку…")
    summary = analyst.summarize(plan, pubmed, trials, stats, settings.trials_for_summary,
                                settings.use_llm, found_methods)
    an_log(f"готово ({summary.model}); ссылок в тексте: {len(summary.cited)}, "
           f"не найдено в данных: {len(summary.unknown_ids)}")

    return RunResult(query, plan, pubmed, trials, stats, summary, found_methods, rounds, chosen, trace, errors)
