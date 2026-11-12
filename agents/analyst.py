"""Агент-аналитик: объединяет результаты других агентов в сводку на русском со ссылками на источники."""
import re
from dataclasses import dataclass, field

from . import llm
from .methods import MethodsResult
from .planner import SearchPlan
from .pubmed_agent import PubMedResult
from .stats_agent import STATUS_RU, StatsResult
from .trials_agent import TrialsResult

SYSTEM_PROMPT = """Ты — агент-аналитик в учебной системе сбора научной информации по психологии и психиатрии.
Тебе передают результаты других агентов: статьи из PubMed, клинические испытания из ClinicalTrials.gov и статистику.
Напиши сводку на русском языке для студента-психолога. Правила:
- Используй ТОЛЬКО переданные данные, ничего не придумывай.
- После каждого утверждения о статье ставь ссылку в квадратных скобках: [PMID и номер статьи из данных],
  об испытании — [номер испытания из данных, начинается с NCT]. Ссылайся только на номера, которые есть в данных ниже.
- Если найденные данные не относятся к теме запроса, честно так и напиши.
- В разделе «Что изучается сейчас» используй данные о методах лечения из статей: сравни, о каких методах пишут
  в статьях и по каким сейчас идут испытания. Не пиши в тексте служебные названия блоков данных.
- Не давай медицинских советов и не рекомендуй лечение конкретному человеку.
- Пиши понятно, без лишнего жаргона. Объём — 300–500 слов.
Структура (markdown, заголовки ###):
### Коротко
### Что говорят опубликованные исследования
### Что изучается сейчас
### Психотерапия или лекарства
### Пробелы и ограничения"""


@dataclass
class Summary:
    text: str
    model: str
    used_llm: bool
    cited: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)  # ссылки, которых нет в данных (галлюцинации)
    error: str = ""


def _methods_lines(found: MethodsResult | None) -> list[str]:
    lines = []
    for m in (found.methods if found else []):
        refs = ", ".join(f"PMID {p}" for p in m.pmids[:3])
        total = "?" if m.trials_total is None else m.trials_total
        active = "?" if m.trials_active is None else m.trials_active
        lines.append(f"- {m.name_ru} ({m.kind}): статей в выборке {len(m.pmids)} [{refs}]; "
                     f"испытаний всего {total}, идут сейчас {active}")
    return lines


def _context(plan: SearchPlan, pubmed: PubMedResult | None, trials: TrialsResult | None,
             stats: StatsResult, n_trials: int, found: MethodsResult | None = None) -> str:
    parts = [f"Тема: {plan.topic_ru}", f"Статистика: {stats.facts}", "", "СТАТЬИ PUBMED:"]
    for a in (pubmed.articles if pubmed else []):
        types = ", ".join(t for t in a.pub_types if t != "Journal Article")
        parts.append(f"[PMID {a.pmid}] ({a.year}; {types}) {a.title}\nАннотация: {a.abstract[:900]}")
    parts += ["", "КЛИНИЧЕСКИЕ ИСПЫТАНИЯ:"]
    for t in (trials.trials[:n_trials] if trials else []):
        parts.append(
            f"[{t.nct_id}] {t.title} | статус: {t.status} | фазы: {', '.join(t.phases) or '—'} | "
            f"вмешательства: {'; '.join(t.interventions[:4]) or '—'} ({', '.join(t.intervention_types)}) | "
            f"страны: {', '.join(t.countries[:4]) or '—'} | участников: {t.enrollment or '—'}"
        )
    if found and found.methods:
        parts += ["", "Методы лечения из статей и число испытаний по ним (агент PubMed нашёл методы, агент ClinicalTrials посчитал испытания):"]
        parts += _methods_lines(found)
    return "\n".join(parts)


def _link_citations(text: str) -> str:
    # списки в скобках «[PMID 1, PMID 2]» или «[NCT…, NCT…]»: убираем скобки, номера станут ссылками ниже
    text = re.sub(r"\[((?:\s*(?:PMID[:\s]*\d+|NCT\d{8})\s*[,;]){1,}\s*(?:PMID[:\s]*\d+|NCT\d{8})\s*)\]",
                  r"\1", text)
    text = re.sub(r"\[PMID[:\s]*(\d+)\]", r"[PMID \1](https://pubmed.ncbi.nlm.nih.gov/\1/)", text)
    text = re.sub(r"\[(NCT\d{8})\]", r"[\1](https://clinicaltrials.gov/study/\1)", text)
    # номера без квадратных скобок, например «(NCT05291533, n=29)», тоже делаем ссылками
    text = re.sub(r"(?<![\[/])\b(NCT\d{8})\b(?!\])", r"[\1](https://clinicaltrials.gov/study/\1)", text)
    text = re.sub(r"(?<!\[)\bPMID[:\s]*(\d+)\b(?!\])", r"[PMID \1](https://pubmed.ncbi.nlm.nih.gov/\1/)", text)
    return text


def _clean_labels(text: str) -> str:
    """Убираем служебные подписи, если модель всё-таки переписала их из данных."""
    text = re.sub(r"\*{0,2}(МЕТОДЫ ИЗ СТАТЕЙ|СТАТЬИ PUBMED|КЛИНИЧЕСКИЕ ИСПЫТАНИЯ)\*{0,2}:\s*", "", text)
    return re.sub(r"\s*\[статистика\]", "", text)


def _check_citations(text: str, pubmed: PubMedResult | None, trials: TrialsResult | None) -> tuple[list, list]:
    known = {a.pmid for a in (pubmed.articles if pubmed else [])} | {t.nct_id for t in (trials.trials if trials else [])}
    cited = re.findall(r"PMID[:\s]*(\d+)", text) + re.findall(r"(NCT\d{8})", text)
    cited = list(dict.fromkeys(cited))
    return cited, [c for c in cited if c not in known]


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _template_summary(plan: SearchPlan, pubmed: PubMedResult | None, trials: TrialsResult | None,
                      stats: StatsResult, found: MethodsResult | None = None) -> str:
    f = stats.facts
    lines = [f"### Коротко\nТема: **{plan.topic_ru}**."]
    if pubmed:
        lines.append(f"В PubMed найдено **{_fmt(f.get('pubmed_total', 0))}** публикаций.")
        if f.get("pubmed_cagr_pct") is not None:
            lines.append(f"За {f['pubmed_period']} число публикаций в среднем менялось на "
                         f"**{f['pubmed_cagr_pct']}% в год**.")
    if trials:
        lines.append(f"В ClinicalTrials.gov — **{_fmt(f.get('trials_total', 0))}** испытаний, "
                     f"из них активных в выборке — **{f.get('trials_active_share', 0)}%**.")
    lines.append("\n### Самые релевантные статьи")
    for a in (pubmed.articles[:5] if pubmed else []):
        lines.append(f"- {a.title} ({a.year}) [PMID {a.pmid}]")
    lines.append("\n### Испытания")
    for t in (trials.trials[:5] if trials else []):
        lines.append(f"- {t.title} — {STATUS_RU.get(t.status, t.status)} [{t.nct_id}]")
    if found and found.methods:
        lines.append("\n### Методы из статей и их испытания")
        lines += _methods_lines(found)
    if "behavioral_share" in f:
        b, d = f["behavioral_share"], f["drug_share"]
        lines.append(f"\n### Психотерапия или лекарства\nПоведенческие вмешательства: **{b['pct']}%** "
                     f"(95% ДИ {b['ci_low']}–{b['ci_high']}%), лекарства: **{d['pct']}%** "
                     f"(95% ДИ {d['ci_low']}–{d['ci_high']}%), n = {b['n']}.")
    lines.append("\n*Сводка собрана по шаблону: нейросеть не подключена, недоступна или не введён пароль.*")
    return "\n".join(lines)


def summarize(plan: SearchPlan, pubmed: PubMedResult | None, trials: TrialsResult | None,
              stats: StatsResult, n_trials: int = 15, use_llm: bool = True,
              found: MethodsResult | None = None) -> Summary:
    if not (pubmed and pubmed.articles) and not (trials and trials.trials):
        return Summary("Нет найденных данных, поэтому сводку составить не из чего.", "—", False)
    error = ""
    if use_llm and llm.is_configured():
        try:
            text, model = llm.chat(SYSTEM_PROMPT, _context(plan, pubmed, trials, stats, n_trials, found))
            cited, unknown = _check_citations(text, pubmed, trials)
            return Summary(_link_citations(_clean_labels(text)), model, True, cited, unknown)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    text = _template_summary(plan, pubmed, trials, stats, found)
    cited, unknown = _check_citations(text, pubmed, trials)
    return Summary(_link_citations(text), "шаблон", False, cited, unknown, error)
