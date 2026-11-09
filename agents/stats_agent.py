"""Агент-статистик: описательная статистика по найденным статьям и испытаниям."""
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .pubmed_agent import PubMedResult
from .trials_agent import ACTIVE_STATUSES, TrialsResult

STATUS_RU = {
    "RECRUITING": "Идёт набор",
    "NOT_YET_RECRUITING": "Набор ещё не начат",
    "ACTIVE_NOT_RECRUITING": "Идёт, набор закрыт",
    "ENROLLING_BY_INVITATION": "Набор по приглашению",
    "COMPLETED": "Завершено",
    "TERMINATED": "Прекращено досрочно",
    "WITHDRAWN": "Отозвано",
    "SUSPENDED": "Приостановлено",
    "UNKNOWN": "Статус неизвестен",
}

INTERVENTION_RU = {
    "BEHAVIORAL": "Поведенческие (психотерапия, тренинги)",
    "DRUG": "Лекарства",
    "DEVICE": "Устройства (стимуляция, приложения)",
    "PROCEDURE": "Процедуры",
    "DIETARY_SUPPLEMENT": "БАДы и питание",
    "BIOLOGICAL": "Биопрепараты",
    "DIAGNOSTIC_TEST": "Диагностика",
    "RADIATION": "Облучение",
    "GENETIC": "Генетические",
    "COMBINATION_PRODUCT": "Комбинированные",
    "OTHER": "Другое",
}

PHASE_RU = {
    "EARLY_PHASE1": "Ранняя фаза 1",
    "PHASE1": "Фаза 1",
    "PHASE2": "Фаза 2",
    "PHASE3": "Фаза 3",
    "PHASE4": "Фаза 4",
    "NA": "Без фазы (не лекарства)",
}

EVIDENCE_TYPES = ["Meta-Analysis", "Systematic Review", "Randomized Controlled Trial", "Review", "Clinical Trial"]


@dataclass
class StatsResult:
    years: pd.DataFrame = field(default_factory=pd.DataFrame)
    statuses: pd.DataFrame = field(default_factory=pd.DataFrame)
    interventions: pd.DataFrame = field(default_factory=pd.DataFrame)
    phases: pd.DataFrame = field(default_factory=pd.DataFrame)
    countries: pd.DataFrame = field(default_factory=pd.DataFrame)
    start_years: pd.DataFrame = field(default_factory=pd.DataFrame)
    facts: dict = field(default_factory=dict)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% доверительный интервал Уилсона для доли."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def _counter_df(counter: Counter, labels: dict | None, name: str, top: int | None = None) -> pd.DataFrame:
    items = counter.most_common(top)
    df = pd.DataFrame(items, columns=[name, "Количество"])
    if labels:
        df[name] = df[name].map(lambda x: labels.get(x, x))
    return df


def analyze(pubmed: PubMedResult | None, trials: TrialsResult | None) -> StatsResult:
    res = StatsResult()
    facts: dict = {}
    this_year = date.today().year

    # ---------- PubMed: динамика публикаций ----------
    if pubmed and pubmed.year_counts:
        years = pd.DataFrame(sorted(pubmed.year_counts.items()), columns=["Год", "Публикаций"])
        years["Неполный год"] = years["Год"] >= this_year
        res.years = years

        full = years[~years["Неполный год"]]
        if len(full) >= 3:
            slope, _ = np.polyfit(full["Год"], full["Публикаций"], 1)
            first, last = full.iloc[0], full.iloc[-1]
            n_years = int(last["Год"] - first["Год"])
            cagr = (last["Публикаций"] / first["Публикаций"]) ** (1 / n_years) - 1 if first["Публикаций"] > 0 else None
            facts["pubmed_trend_slope"] = round(float(slope), 1)
            facts["pubmed_cagr_pct"] = round(float(cagr) * 100, 1) if cagr is not None else None
            facts["pubmed_period"] = f"{int(first['Год'])}–{int(last['Год'])}"
            facts["pubmed_first_last"] = (int(first["Публикаций"]), int(last["Публикаций"]))

    if pubmed:
        facts["pubmed_total"] = pubmed.total_count
        types = Counter(t for a in pubmed.articles for t in a.pub_types if t in EVIDENCE_TYPES)
        facts["top_articles_types"] = dict(types)

    # ---------- ClinicalTrials ----------
    if trials and trials.trials:
        ts = trials.trials
        n = len(ts)
        facts["trials_total"] = trials.total_count
        facts["trials_sample"] = n

        res.statuses = _counter_df(Counter(t.status for t in ts), STATUS_RU, "Статус")
        res.interventions = _counter_df(
            Counter(x for t in ts for x in t.intervention_types), INTERVENTION_RU, "Тип вмешательства"
        )
        res.phases = _counter_df(Counter(p for t in ts for p in t.phases), PHASE_RU, "Фаза")
        res.countries = _counter_df(Counter(c for t in ts for c in t.countries), None, "Страна", top=12)
        res.start_years = (
            pd.Series([t.start_year for t in ts if t.start_year]).value_counts().sort_index()
            .rename_axis("Год старта").reset_index(name="Испытаний")
        )

        active = sum(t.status in ACTIVE_STATUSES for t in ts)
        facts["trials_active_share"] = round(active / n * 100, 1)

        # психотерапия vs лекарства среди интервенционных испытаний
        interventional = [t for t in ts if t.intervention_types]
        m = len(interventional)
        if m:
            beh = sum("BEHAVIORAL" in t.intervention_types for t in interventional)
            drug = sum("DRUG" in t.intervention_types for t in interventional)
            lo, hi = wilson_ci(beh, m)
            facts["behavioral_share"] = {
                "pct": round(beh / m * 100, 1), "ci_low": round(lo * 100, 1), "ci_high": round(hi * 100, 1), "n": m,
            }
            lo, hi = wilson_ci(drug, m)
            facts["drug_share"] = {
                "pct": round(drug / m * 100, 1), "ci_low": round(lo * 100, 1), "ci_high": round(hi * 100, 1), "n": m,
            }

        enroll = [t.enrollment for t in ts if t.enrollment]
        if enroll:
            q1, med, q3 = np.percentile(enroll, [25, 50, 75])
            facts["enrollment"] = {"median": int(med), "q1": int(q1), "q3": int(q3), "n": len(enroll)}

        if not res.countries.empty:
            facts["top_countries"] = res.countries.head(5)["Страна"].tolist()

    res.facts = facts
    return res
