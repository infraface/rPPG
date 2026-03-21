#!/usr/bin/env python3
"""
fetch_papers.py — Daily arXiv paper fetcher for rPPG Research Hub

Fetches new papers from cs.CV and cs.AI, scores relevance to rPPG research,
and outputs:
  - data/daily_papers.json   (today's papers, loaded by index.html)
  - data/archive/YYYY-MM-DD.json  (daily snapshot for the archive page)

Relevance scoring rules:
  ═══════════════════════════════════════════════════════════════
  Each paper gets a relevance score (0–100) based on keyword matches
  in the title + abstract. The scoring system has three tiers:

  TIER 1 — Core rPPG keywords (weight: 15 each, max contribution: 60)
    Direct mentions of the technology or its variants.

  TIER 2 — Related task keywords (weight: 6 each, max contribution: 30)
    Physiological signals, face analysis, health monitoring concepts
    that are strongly associated with rPPG research.

  TIER 3 — Peripheral keywords (weight: 3 each, max contribution: 15)
    Broader ML/CV concepts that are relevant but not specific.

  Final score = min(100, tier1 + tier2 + tier3)
  Papers with score < RELEVANCE_THRESHOLD (default: 12) are dropped.
  ═══════════════════════════════════════════════════════════════

Usage:
  python fetch_papers.py                  # fetch today's papers
  python fetch_papers.py --date 2025-03-20  # fetch for a specific date

Requires: requests, feedparser
  pip install requests feedparser
"""

import json
import os
import re
import sys
import argparse
import feedparser
from datetime import datetime, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════
# ★ EDITABLE: Relevance keywords and weights
# ═══════════════════════════════════════════════════════

TIER1_KEYWORDS = [
    # Core rPPG terms — highest relevance
    "rppg", "remote photoplethysmography", "remote ppg",
    "video-based heart rate", "camera-based heart rate",
    "contactless heart rate", "non-contact heart rate",
    "video-based physiological", "camera-based physiological",
    "remote physiological measurement", "remote vital sign",
    "contactless vital sign",
]

TIER2_KEYWORDS = [
    # Strongly related tasks
    "photoplethysmography", "ppg", "heart rate estimation",
    "heart rate variability", "hrv", "blood volume pulse", "bvp",
    "pulse rate", "blood oxygen", "spo2", "respiratory rate",
    "facial video", "face video", "skin color change",
    "physiological signal", "vital sign monitoring",
    "euler video magnification", "video magnification",
    "atrial fibrillation detection", "cardiac", "cardiopulmonary",
    "face de-identification", "face privacy",
    "deepfake detection", "face forgery",
]

TIER3_KEYWORDS = [
    # Peripheral but relevant concepts
    "self-supervised video", "video transformer",
    "temporal attention", "spatio-temporal",
    "skin segmentation", "face detection", "face recognition",
    "action unit", "facial action", "facial expression",
    "health monitoring", "wearable", "telemedicine",
    "signal processing", "blind source separation",
    "independent component analysis",
    "optical flow", "motion estimation",
    "domain adaptation", "domain generalization",
    "transfer learning", "contrastive learning",
    "noise robust", "motion artifact",
]

TIER1_WEIGHT = 15
TIER2_WEIGHT = 6
TIER3_WEIGHT = 3
TIER1_CAP = 60
TIER2_CAP = 30
TIER3_CAP = 15

RELEVANCE_THRESHOLD = 12   # minimum score to include
MAX_PAPERS_PER_DAY = 30    # cap output size

# ═══════════════════════════════════════════════════════
# Scoring engine
# ═══════════════════════════════════════════════════════

def compute_relevance(title: str, abstract: str) -> dict:
    """Compute relevance score and matched keywords."""
    text = (title + " " + abstract).lower()

    t1_matches = [kw for kw in TIER1_KEYWORDS if kw.lower() in text]
    t2_matches = [kw for kw in TIER2_KEYWORDS if kw.lower() in text]
    t3_matches = [kw for kw in TIER3_KEYWORDS if kw.lower() in text]

    t1_score = min(len(t1_matches) * TIER1_WEIGHT, TIER1_CAP)
    t2_score = min(len(t2_matches) * TIER2_WEIGHT, TIER2_CAP)
    t3_score = min(len(t3_matches) * TIER3_WEIGHT, TIER3_CAP)

    score = min(100, t1_score + t2_score + t3_score)

    # Determine relevance tier label
    if t1_score > 0:
        tier = "core"
    elif t2_score >= 12:
        tier = "high"
    elif t2_score > 0:
        tier = "medium"
    else:
        tier = "low"

    return {
        "score": score,
        "tier": tier,
        "matched_keywords": t1_matches + t2_matches + t3_matches,
    }


# ═══════════════════════════════════════════════════════
# arXiv fetcher
# ═══════════════════════════════════════════════════════

ARXIV_RSS_URLS = [
    "https://rss.arxiv.org/rss/cs.CV",
    "https://rss.arxiv.org/rss/cs.AI",
]

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def extract_arxiv_id(link: str) -> str:
    """Extract arXiv ID from a URL like http://arxiv.org/abs/2503.12345"""
    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', link)
    return match.group(1) if match else ""


def fetch_from_rss() -> list[dict]:
    """Fetch today's papers from arXiv RSS feeds."""
    seen_ids = set()
    papers = []

    for url in ARXIV_RSS_URLS:
        print(f"  Fetching RSS: {url}")
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"  ⚠ Failed to fetch {url}: {e}")
            continue

        for entry in feed.entries:
            arxiv_id = extract_arxiv_id(entry.get("link", ""))
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            title = entry.get("title", "").strip()
            # RSS descriptions can contain HTML; strip tags
            abstract = re.sub(r'<[^>]+>', '', entry.get("summary", "")).strip()

            # Some RSS entries have minimal abstracts; skip if too short
            if len(abstract) < 50:
                abstract = title  # fallback

            authors_raw = entry.get("author", entry.get("dc_creator", ""))
            if isinstance(authors_raw, str):
                authors = [a.strip() for a in authors_raw.split(",")]
            else:
                authors = [authors_raw]

            # Determine primary category from tags
            categories = []
            for tag in entry.get("tags", []):
                term = tag.get("term", "")
                if term:
                    categories.append(term)

            papers.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5],  # cap to first 5
                "categories": categories,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            })

    print(f"  Fetched {len(papers)} unique papers from RSS")
    return papers


def fetch_from_api(date_str: str) -> list[dict]:
    """Fallback: fetch papers via arXiv API for a specific date range."""
    import requests
    import time

    # Search for papers submitted on the given date
    # arXiv API uses submittedDate field
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = dt.strftime("%Y%m%d") + "0000"
    end = dt.strftime("%Y%m%d") + "2359"

    papers = []
    seen_ids = set()

    for cat in ["cs.CV", "cs.AI"]:
        query = f"cat:{cat} AND submittedDate:[{start} TO {end}]"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 200,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        print(f"  API query: {cat} for {date_str}")
        try:
            resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
            feed = feedparser.parse(resp.text)
        except Exception as e:
            print(f"  ⚠ API request failed: {e}")
            continue

        for entry in feed.entries:
            arxiv_id = extract_arxiv_id(entry.get("id", ""))
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            title = re.sub(r'\s+', ' ', entry.get("title", "")).strip()
            abstract = re.sub(r'\s+', ' ', entry.get("summary", "")).strip()
            authors = [a.get("name", "") for a in entry.get("authors", [])]

            categories = [t.get("term", "") for t in entry.get("tags", [])]

            papers.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5],
                "categories": categories,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            })

        time.sleep(3)  # be polite to arXiv API

    print(f"  Fetched {len(papers)} papers via API")
    return papers


# ═══════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════

def process_papers(papers: list[dict]) -> list[dict]:
    """Score, filter, and sort papers by relevance."""
    scored = []
    for p in papers:
        rel = compute_relevance(p["title"], p["abstract"])
        if rel["score"] >= RELEVANCE_THRESHOLD:
            p["relevance_score"] = rel["score"]
            p["relevance_tier"] = rel["tier"]
            p["matched_keywords"] = rel["matched_keywords"]
            scored.append(p)

    # Sort by score descending
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:MAX_PAPERS_PER_DAY]


def save_output(papers: list[dict], date_str: str):
    """Save to data/daily_papers.json and data/archive/YYYY-MM-DD.json"""
    data_dir = Path("data")
    archive_dir = data_dir / "archive"
    data_dir.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)

    output = {
        "date": date_str,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "paper_count": len(papers),
        "papers": papers,
    }

    # Daily file (loaded by index.html)
    daily_path = data_dir / "daily_papers.json"
    with open(daily_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(papers)} papers → {daily_path}")

    # Archive file
    archive_path = archive_dir / f"{date_str}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Archived → {archive_path}")

    # Clean up archives older than 90 days
    cutoff = datetime.now() - timedelta(days=90)
    removed = 0
    for f in archive_dir.glob("*.json"):
        try:
            fdate = datetime.strptime(f.stem, "%Y-%m-%d")
            if fdate < cutoff:
                f.unlink()
                removed += 1
        except ValueError:
            pass
    if removed:
        print(f"  🗑 Removed {removed} archive files older than 90 days")


def build_archive_index():
    """Generate data/archive_index.json listing all available archive dates."""
    archive_dir = Path("data/archive")
    if not archive_dir.exists():
        return

    dates = []
    for f in sorted(archive_dir.glob("*.json"), reverse=True):
        try:
            # Read paper count from each file
            with open(f, "r") as fh:
                d = json.load(fh)
                dates.append({
                    "date": f.stem,
                    "paper_count": d.get("paper_count", 0),
                })
        except Exception:
            dates.append({"date": f.stem, "paper_count": 0})

    index_path = Path("data/archive_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"dates": dates}, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Archive index: {len(dates)} dates → {index_path}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fetch rPPG-related papers from arXiv")
    parser.add_argument("--date", type=str, default=None,
                        help="Fetch papers for a specific date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--use-api", action="store_true",
                        help="Use arXiv API instead of RSS (slower, for backfill).")
    args = parser.parse_args()

    date_str = args.date or datetime.utcnow().strftime("%Y-%m-%d")
    print(f"═══ rPPG Paper Fetcher — {date_str} ═══")

    if args.use_api:
        raw_papers = fetch_from_api(date_str)
    else:
        raw_papers = fetch_from_rss()

    scored_papers = process_papers(raw_papers)
    save_output(scored_papers, date_str)
    build_archive_index()

    print(f"═══ Done. {len(scored_papers)} relevant papers found. ═══")


if __name__ == "__main__":
    main()