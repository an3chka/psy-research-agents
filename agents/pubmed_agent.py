"""Агент PubMed: ищет статьи через NCBI E-utilities и считает динамику публикаций по годам."""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from .http import RateLimiter, make_session

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "psy-research-agents"

_session = make_session()
# NCBI разрешает 3 запроса/с без ключа и 10 — с ключом
_limiter = RateLimiter(9 if os.environ.get("NCBI_API_KEY") else 2.5)


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: int | None
    authors: str
    pub_types: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass
class PubMedResult:
    query_used: str
    total_count: int
    articles: list[Article]
    year_counts: dict[int, int]


def _get(endpoint: str, **params) -> requests.Response:
    params = {"tool": TOOL, **params}
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    _limiter.wait()
    resp = _session.get(f"{EUTILS}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp


def _esearch(term: str, retmax: int, mindate: int | None = None, maxdate: int | None = None) -> tuple[int, list[str]]:
    params = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    if mindate and maxdate:
        params.update(datetype="pdat", mindate=str(mindate), maxdate=str(maxdate))
    data = _get("esearch.fcgi", **params).json()["esearchresult"]
    return int(data.get("count", 0)), data.get("idlist", [])


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _text(node: ET.Element | None) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def _parse_year(article: ET.Element) -> int | None:
    for path in (".//PubDate/Year", ".//ArticleDate/Year", ".//PubDate/MedlineDate"):
        value = _text(article.find(path))
        if value[:4].isdigit():
            return int(value[:4])
    return None


def _efetch(pmids: list[str]) -> list[Article]:
    if not pmids:
        return []
    xml = _get("efetch.fcgi", db="pubmed", id=",".join(pmids), retmode="xml").content
    root = ET.fromstring(xml)
    by_id = {}
    for node in root.findall(".//PubmedArticle"):
        pmid = _text(node.find(".//PMID"))
        abstract_parts = []
        for part in node.findall(".//Abstract/AbstractText"):
            label = part.get("Label")
            text = _text(part)
            abstract_parts.append(f"{label}: {text}" if label else text)
        authors = []
        for a in node.findall(".//AuthorList/Author")[:3]:
            last, initials = _text(a.find("LastName")), _text(a.find("Initials"))
            if last:
                authors.append(f"{last} {initials}".strip())
        if len(node.findall(".//AuthorList/Author")) > 3:
            authors.append("et al.")
        by_id[pmid] = Article(
            pmid=pmid,
            title=_text(node.find(".//ArticleTitle")),
            abstract=" ".join(abstract_parts),
            journal=_text(node.find(".//Journal/Title")),
            year=_parse_year(node),
            authors=", ".join(authors),
            pub_types=[_text(p) for p in node.findall(".//PublicationType")],
        )
    # сохраняем порядок релевантности из esearch
    return [by_id[p] for p in pmids if p in by_id]


def search(
    pubmed_query: str,
    fallback_query: str,
    max_articles: int = 15,
    year_from: int = 2015,
    year_to: int = 2026,
    with_trend: bool = True,
    log=lambda msg: None,
) -> PubMedResult:
    query = pubmed_query
    total, ids = _esearch(query, max_articles, year_from, year_to)
    if total == 0 and fallback_query and fallback_query != pubmed_query:
        log(f"по запросу «{pubmed_query}» ничего не нашлось, пробую упрощённый: «{fallback_query}»")
        query = fallback_query
        total, ids = _esearch(query, max_articles, year_from, year_to)
    log(f"найдено {_fmt(total)} статей за {year_from}–{year_to}, загружаю {len(ids)} самых релевантных")

    articles = _efetch(ids)

    year_counts: dict[int, int] = {}
    if with_trend and total > 0:
        log("считаю число публикаций по каждому году")
        for year in range(year_from, year_to + 1):
            year_counts[year], _ = _esearch(query, 0, year, year)

    return PubMedResult(query_used=query, total_count=total, articles=articles, year_counts=year_counts)
