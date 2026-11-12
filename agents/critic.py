"""Агент-критик: проверяет, отвечают ли найденные данные на вопрос пользователя.

Если не отвечают, критик объясняет, что не так, и оркестратор просит планировщика
переписать запросы с учётом этого замечания (обратная связь между агентами).
"""
from dataclasses import dataclass

from . import llm
from .planner import SearchPlan
from .pubmed_agent import PubMedResult
from .trials_agent import TrialsResult

PASS_SCORE = 6  # оценка от 0 до 10, с которой результат считается подходящим

SYSTEM_PROMPT = """Ты — агент-критик в системе поиска научной информации по психологии и психиатрии.
Тебе дают вопрос пользователя, поисковые запросы и названия первых найденных статей и клинических испытаний.
Оцени по шкале от 0 до 10, насколько найденное отвечает именно на этот вопрос
(учитывай возраст, группу людей, метод, состояние, если они есть в вопросе).
Ответь ТОЛЬКО JSON-объектом без пояснений:
{"score": 0-10,
 "problem": "что не так с результатами, одно короткое предложение по-русски; пусто, если всё хорошо",
 "advice": "как исправить поисковые запросы, одно короткое предложение по-русски; пусто, если всё хорошо"}"""


@dataclass
class Review:
    score: int | None  # 0–10; None, если нейросеть не оценивала
    problem: str
    advice: str
    model: str

    @property
    def ok(self) -> bool:
        return self.score is None or self.score >= PASS_SCORE


def _titles(pubmed: PubMedResult | None, trials: TrialsResult | None, n: int = 8) -> str:
    lines = [f"Найдено статей в PubMed: {pubmed.total_count if pubmed else 0}"]
    lines += [f"- {a.title}" for a in (pubmed.articles[:n] if pubmed else [])]
    lines.append(f"Найдено испытаний в ClinicalTrials.gov: {trials.total_count if trials else 0}")
    lines += [f"- {t.title}" for t in (trials.trials[:n] if trials else [])]
    return "\n".join(lines)


def review(query: str, plan: SearchPlan, pubmed: PubMedResult | None, trials: TrialsResult | None,
           use_llm: bool = True) -> Review:
    n_found = (len(pubmed.articles) if pubmed else 0) + (len(trials.trials) if trials else 0)
    if n_found == 0:
        return Review(0, "ничего не найдено ни в одном источнике",
                      "сделать запросы шире: убрать лишние условия и редкие термины", "правила")
    if plan.source == "preset":
        # готовые темы проверены заранее — не тратим запрос к нейросети
        return Review(None, "", "", "готовая тема, запросы проверены заранее")
    if not (use_llm and llm.is_configured()):
        return Review(None, "", "", "без нейросети")
    user = (f"Вопрос пользователя: {query}\n"
            f"Запрос PubMed: {plan.pubmed_query}\n"
            f"Запрос ClinicalTrials: condition={plan.trials_condition!r}, term={plan.trials_term!r}\n\n"
            + _titles(pubmed, trials))
    try:
        text, model = llm.chat(SYSTEM_PROMPT, user, temperature=0.0, max_tokens=400)
        data = llm.extract_json(text)
        score = max(0, min(10, int(round(float(data.get("score", 10))))))
        return Review(score, str(data.get("problem") or ""), str(data.get("advice") or ""), model)
    except Exception as exc:  # noqa: BLE001 — критик не должен ломать весь поиск
        return Review(None, "", "", f"нейросеть не ответила ({type(exc).__name__})")
