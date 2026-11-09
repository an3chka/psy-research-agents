"""Агент ClinicalTrials.gov: ищет клинические испытания через API v2."""
from dataclasses import dataclass

from .http import make_session

API = "https://clinicaltrials.gov/api/v2/studies"
_session = make_session()

FIELDS = ",".join([
    "NCTId", "BriefTitle", "OverallStatus", "Phase", "StudyType",
    "InterventionType", "InterventionName", "Condition", "LocationCountry",
    "EnrollmentCount", "StartDate", "BriefSummary", "LeadSponsorName",
])

ACTIVE_STATUSES = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]


@dataclass
class Trial:
    nct_id: str
    title: str
    status: str
    phases: list[str]
    study_type: str
    intervention_types: list[str]
    interventions: list[str]
    conditions: list[str]
    countries: list[str]
    enrollment: int | None
    start_year: int | None
    summary: str
    sponsor: str

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"


@dataclass
class TrialsResult:
    total_count: int
    trials: list[Trial]  # выборка для статистики (до sample_size), первые — самые релевантные


def _parse(study: dict) -> Trial:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    arms = p.get("armsInterventionsModule", {})
    interventions = arms.get("interventions", []) or []
    locations = p.get("contactsLocationsModule", {}).get("locations", []) or []
    start = (status.get("startDateStruct") or {}).get("date", "")
    enrollment = (design.get("enrollmentInfo") or {}).get("count")
    return Trial(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        status=status.get("overallStatus", "UNKNOWN"),
        phases=design.get("phases", []) or [],
        study_type=design.get("studyType", ""),
        intervention_types=sorted({i.get("type", "OTHER") for i in interventions}),
        interventions=[i.get("name", "") for i in interventions],
        conditions=p.get("conditionsModule", {}).get("conditions", []) or [],
        countries=sorted({loc.get("country") for loc in locations if loc.get("country")}),
        enrollment=int(enrollment) if enrollment is not None else None,
        start_year=int(start[:4]) if start[:4].isdigit() else None,
        summary=p.get("descriptionModule", {}).get("briefSummary", ""),
        sponsor=p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
    )


def search(
    condition: str,
    term: str,
    only_active: bool = False,
    sample_size: int = 200,
    year_from: int | None = None,
    log=lambda msg: None,
) -> TrialsResult:
    params = {"format": "json", "countTotal": "true", "fields": FIELDS, "pageSize": min(sample_size, 1000)}
    if condition:
        params["query.cond"] = condition
    if term:
        params["query.term"] = term
    if not condition and not term:
        raise ValueError("Пустой запрос для ClinicalTrials.gov")
    if only_active:
        params["filter.overallStatus"] = ",".join(ACTIVE_STATUSES)
    if year_from:
        params["filter.advanced"] = f"AREA[StartDate]RANGE[{year_from}-01-01,MAX]"

    resp = _session.get(API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    trials = [_parse(s) for s in data.get("studies", [])]
    total = int(data.get("totalCount", len(trials)))
    total_txt = f"{total:,}".replace(",", " ")
    log(f"найдено {total_txt} испытаний, для анализа загружено {len(trials)}")
    return TrialsResult(total_count=total, trials=trials)
