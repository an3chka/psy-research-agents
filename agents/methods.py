"""Передача данных между агентами: методы лечения из статей → проверка в клинических испытаниях.

1. Агент PubMed выделяет в найденных статьях, какие методы помощи изучают
   (нейросетью, а без неё — по словарю известных методов).
2. Агент ClinicalTrials для каждого метода считает, сколько испытаний по нему
   зарегистрировано и сколько идёт прямо сейчас.
Получается связка «что уже опубликовано → что проверяют сейчас».
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import llm, trials_agent
from .pubmed_agent import Article

MAX_METHODS = 6

SYSTEM_PROMPT = """Ты — помощник агента PubMed в системе поиска научной информации по психологии и психиатрии.
Тебе дают найденные статьи (PMID, название, аннотация). Выдели до 6 методов помощи или лечения
(психотерапия, лекарства, цифровые программы, стимуляция мозга и т. п.), которые изучаются в этих статьях.
Не включай диагнозы, опросники и методы статистики.
Ответь ТОЛЬКО JSON-объектом без пояснений:
{"methods": [{"name_ru": "общее название метода по-русски, 1–3 слова (например «КПТ», «Майндфулнес», «Антидепрессанты»)",
              "query_en": "простое английское название метода для ClinicalTrials.gov, 1–3 слова, без аббревиатур, скобок и косых черт, без названия болезни (например «antidepressant», «mindfulness», «exercise»)",
              "kind": "психотерапия | лекарство | цифровое | стимуляция | другое",
              "pmids": ["номера статей, где упоминается метод"]}]}"""

# Запасной словарь: (название, запрос для ClinicalTrials, тип, шаблон для поиска в тексте статьи)
KNOWN_METHODS = [
    ("КПТ", "Cognitive Behavioral Therapy", "психотерапия", r"cognitive[- ]behavio|\bCBT"),
    ("Майндфулнес", "Mindfulness", "психотерапия", r"mindfulness|\bMBSR\b|\bMBCT\b"),
    ("Терапия принятия и ответственности", "Acceptance and Commitment Therapy", "психотерапия",
     r"acceptance and commitment"),
    ("Диалектическая поведенческая терапия", "Dialectical Behavior Therapy", "психотерапия",
     r"dialectical behavio|\bDBT\b"),
    ("Межличностная психотерапия", "Interpersonal Psychotherapy", "психотерапия", r"interpersonal (psycho)?therapy"),
    ("Психодинамическая терапия", "Psychodynamic", "психотерапия", r"psychodynamic"),
    ("Экспозиционная терапия", "Exposure Therapy", "психотерапия", r"exposure therapy|prolonged exposure"),
    ("EMDR", "EMDR", "психотерапия", r"\bEMDR\b|eye movement desensiti"),
    ("Психообразование", "Psychoeducation", "психотерапия", r"psychoeducation"),
    ("Онлайн- и мобильные программы", "Internet-based", "цифровое",
     r"internet[- ](based|delivered)|digital|smartphone|mobile app|online (intervention|program|therapy|treatment)"),
    ("Физическая активность", "Exercise", "другое", r"\bexercise|physical activity|\byoga\b"),
    ("Антидепрессанты СИОЗС", "SSRI OR sertraline OR fluoxetine OR escitalopram", "лекарство",
     r"\bSSRIs?\b|serotonin reuptake|sertraline|fluoxetine|escitalopram|paroxetine|citalopram"),
    ("Кетамин и эскетамин", "ketamine OR esketamine", "лекарство", r"ketamine"),
    ("Психоделики", "psilocybin OR MDMA", "лекарство", r"psilocybin|psychedelic|\bMDMA\b"),
    ("Стимуляторы", "methylphenidate OR lisdexamfetamine", "лекарство",
     r"methylphenidate|amphetamine|stimulant medication"),
    ("Мелатонин", "melatonin", "лекарство", r"melatonin"),
    ("ТМС", "Transcranial Magnetic Stimulation", "стимуляция", r"transcranial magnetic|\br?TMS\b"),
    ("Нейро- и биофидбек", "neurofeedback OR biofeedback", "другое", r"neurofeedback|biofeedback"),
]


@dataclass
class Method:
    name_ru: str
    query_en: str
    kind: str
    pmids: list[str] = field(default_factory=list)
    trials_total: int | None = None
    trials_active: int | None = None


@dataclass
class MethodsResult:
    methods: list[Method]
    source: str  # нейросеть или словарь
    error: str = ""


def _by_dictionary(articles: list[Article]) -> list[Method]:
    found = []
    for name, query, kind, pattern in KNOWN_METHODS:
        rx = re.compile(pattern, flags=re.I)
        pmids = [a.pmid for a in articles if rx.search(f"{a.title} {a.abstract}")]
        if pmids:
            found.append(Method(name, query, kind, pmids))
    return sorted(found, key=lambda m: -len(m.pmids))[:MAX_METHODS]


def _by_llm(articles: list[Article]) -> tuple[list[Method], str]:
    known = {a.pmid for a in articles}
    context = "\n".join(f"[PMID {a.pmid}] {a.title}\n{a.abstract[:600]}" for a in articles)
    text, model = llm.chat(SYSTEM_PROMPT, context, temperature=0.0, max_tokens=900)
    methods = []
    for m in llm.extract_json(text).get("methods", [])[:MAX_METHODS]:
        pmids = [str(p).replace("PMID", "").strip() for p in m.get("pmids", [])]
        pmids = [p for p in dict.fromkeys(pmids) if p in known]  # только реально найденные статьи
        name, query = str(m.get("name_ru") or "").strip(), str(m.get("query_en") or "").strip()
        if name and query and pmids:
            methods.append(Method(name, query, str(m.get("kind") or "другое"), pmids))
    return sorted(methods, key=lambda m: -len(m.pmids)), model


def extract(articles: list[Article], use_llm: bool = True) -> MethodsResult:
    """Шаг агента PubMed: какие методы лечения изучаются в найденных статьях."""
    if not articles:
        return MethodsResult([], "—")
    error = ""
    if use_llm and llm.is_configured():
        try:
            methods, model = _by_llm(articles)
            if methods:
                return MethodsResult(methods, f"нейросеть ({model})")
            error = "нейросеть не нашла методов"
        except Exception as exc:  # noqa: BLE001
            error = f"нейросеть не ответила ({type(exc).__name__})"
    return MethodsResult(_by_dictionary(articles), "словарь методов", error)


def check_trials(result: MethodsResult, condition: str, year_from: int | None, log=lambda msg: None) -> None:
    """Шаг агента ClinicalTrials: сколько испытаний по каждому методу (всего и идущих сейчас)."""
    if not result.methods:
        return

    def _count(m: Method):
        try:
            m.trials_total, m.trials_active = trials_agent.count_intervention(condition, m.query_en, year_from)
        except Exception:  # noqa: BLE001 — один метод не должен ломать остальные
            pass

    with ThreadPoolExecutor(max_workers=MAX_METHODS) as pool:
        list(pool.map(_count, result.methods))
    log("проверил методы: " + "; ".join(
        f"{m.name_ru} — {m.trials_total if m.trials_total is not None else '?'} испытаний, "
        f"идут сейчас {m.trials_active if m.trials_active is not None else '?'}" for m in result.methods))
