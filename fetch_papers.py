#!/usr/bin/env python3
"""
fetch_papers.py — Weekly arXiv paper fetcher for rPPG Research Hub

Runs every Friday via GitHub Actions. Collects papers from the past 7 days
from cs.CV and cs.AI, scores relevance to rPPG research.

Outputs:
  - data/weekly_papers.json          (this week's papers, shown on index.html)
  - data/recent_papers.json          (rolling 90-day window, shown on recent page)
  - data/archive/YYYY-MM-DD.json     (weekly snapshot, keyed by Friday date)
  - data/archive_index.json          (index of all archive weeks)

Usage:
  python fetch_papers.py                        # collect past 7 days
  python fetch_papers.py --days 14              # collect past 14 days
  python fetch_papers.py --start 2025-03-10 --end 2025-03-17

Requires: requests feedparser
  pip install requests feedparser
"""

import json, re, argparse, time
import requests, feedparser
from datetime import datetime, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════
# ★ EDITABLE: Relevance keywords and weights
# ═══════════════════════════════════════════════════════

TIER1_KEYWORDS = [
    "rppg", "remote photoplethysmography", "remote ppg",
    "video-based heart rate", "camera-based heart rate",
    "contactless heart rate", "non-contact heart rate",
    "video-based physiological", "camera-based physiological",
    "remote physiological measurement", "remote vital sign",
    "contactless vital sign",
]

TIER2_KEYWORDS = [
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

TIER1_WEIGHT, TIER2_WEIGHT, TIER3_WEIGHT = 15, 6, 3
TIER1_CAP, TIER2_CAP, TIER3_CAP = 60, 30, 15
RELEVANCE_THRESHOLD = 12
MAX_PAPERS_PER_WEEK = 50
RECENT_WINDOW_DAYS = 90
ARCHIVE_RETENTION_DAYS = 1095

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.CV", "cs.AI"]

# ═══════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════

def compute_relevance(title, abstract):
    text = (title + " " + abstract).lower()
    t1 = [kw for kw in TIER1_KEYWORDS if kw.lower() in text]
    t2 = [kw for kw in TIER2_KEYWORDS if kw.lower() in text]
    t3 = [kw for kw in TIER3_KEYWORDS if kw.lower() in text]
    s1 = min(len(t1) * TIER1_WEIGHT, TIER1_CAP)
    s2 = min(len(t2) * TIER2_WEIGHT, TIER2_CAP)
    s3 = min(len(t3) * TIER3_WEIGHT, TIER3_CAP)
    score = min(100, s1 + s2 + s3)
    tier = "core" if s1 > 0 else ("high" if s2 >= 12 else ("medium" if s2 > 0 else "low"))
    return {"score": score, "tier": tier, "matched_keywords": t1 + t2 + t3}

# ═══════════════════════════════════════════════════════
# arXiv API fetcher (7-day range)
# ═══════════════════════════════════════════════════════

def extract_arxiv_id(link):
    m = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', link)
    return m.group(1) if m else ""

def extract_date(entry):
    """Extract publication date from entry."""
    published = entry.get("published", "")
    if published:
        try:
            return datetime.strptime(published[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def fetch_papers_range(start_date, end_date):
    """Fetch papers from arXiv API for a date range."""
    start_str = start_date.strftime("%Y%m%d") + "0000"
    end_str = end_date.strftime("%Y%m%d") + "2359"
    seen_ids = set()
    papers = []

    for cat in CATEGORIES:
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        offset = 0
        batch_size = 200

        while True:
            params = {
                "search_query": query,
                "start": offset,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            print(f"  API: {cat} offset={offset}")
            try:
                resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
                feed = feedparser.parse(resp.text)
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
                break

            if not feed.entries:
                break

            new_count = 0
            for entry in feed.entries:
                arxiv_id = extract_arxiv_id(entry.get("id", ""))
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)
                new_count += 1

                title = re.sub(r'\s+', ' ', entry.get("title", "")).strip()
                abstract = re.sub(r'\s+', ' ', entry.get("summary", "")).strip()
                authors = [a.get("name", "") for a in entry.get("authors", [])]
                categories = [t.get("term", "") for t in entry.get("tags", [])]
                pub_date = extract_date(entry)

                papers.append({
                    "id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors[:20],
                    "categories": categories,
                    "date": pub_date,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                })

            if new_count == 0 or len(feed.entries) < batch_size:
                break
            offset += batch_size
            time.sleep(3)

        time.sleep(3)

    print(f"  Total: {len(papers)} papers from {start_date.date()} to {end_date.date()}")
    return papers

# ═══════════════════════════════════════════════════════
# Processing
# ═══════════════════════════════════════════════════════

def score_and_filter(papers):
    scored = []
    for p in papers:
        rel = compute_relevance(p["title"], p["abstract"])
        if rel["score"] >= RELEVANCE_THRESHOLD:
            p["relevance_score"] = rel["score"]
            p["relevance_tier"] = rel["tier"]
            p["matched_keywords"] = rel["matched_keywords"]
            scored.append(p)
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:MAX_PAPERS_PER_WEEK]

# ═══════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════

def save_weekly(papers, week_start, week_end):
    """Save weekly_papers.json (loaded by index.html)."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output = {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end": week_end.strftime("%Y-%m-%d"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "paper_count": len(papers),
        "papers": papers,
    }

    path = data_dir / "weekly_papers.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Weekly: {len(papers)} papers → {path}")
    return output

def save_archive(weekly_output, week_end):
    """Save weekly snapshot to archive."""
    archive_dir = Path("data/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"{week_end.strftime('%Y-%m-%d')}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(weekly_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Archive: → {archive_path}")

    # Clean old archives
    cutoff = datetime.now() - timedelta(days=ARCHIVE_RETENTION_DAYS)
    removed = 0
    for fp in archive_dir.glob("*.json"):
        try:
            if datetime.strptime(fp.stem, "%Y-%m-%d") < cutoff:
                fp.unlink()
                removed += 1
        except ValueError:
            pass
    if removed:
        print(f"  🗑 Removed {removed} archives older than {ARCHIVE_RETENTION_DAYS} days")

def build_recent(days=RECENT_WINDOW_DAYS):
    """Merge recent archive files into recent_papers.json (past 90 days)."""
    archive_dir = Path("data/archive")
    if not archive_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=days)
    all_papers = []
    weeks = []

    for fp in sorted(archive_dir.glob("*.json"), reverse=True):
        try:
            fdate = datetime.strptime(fp.stem, "%Y-%m-%d")
            if fdate < cutoff:
                continue
        except ValueError:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        week_papers = data.get("papers", [])
        weeks.append({
            "week_end": fp.stem,
            "week_start": data.get("week_start", ""),
            "paper_count": len(week_papers),
        })
        all_papers.extend(week_papers)

    # Deduplicate by paper ID
    seen = set()
    unique = []
    for p in all_papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    # Sort by date descending, then score descending
    unique.sort(key=lambda x: (x.get("date", ""), x.get("relevance_score", 0)), reverse=True)

    output = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": days,
        "paper_count": len(unique),
        "weeks": weeks,
        "papers": unique,
    }

    path = Path("data/recent_papers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Recent: {len(unique)} papers (past {days} days) → {path}")

def build_archive_index():
    """Generate archive_index.json listing all archived weeks."""
    archive_dir = Path("data/archive")
    if not archive_dir.exists():
        return

    entries = []
    for fp in sorted(archive_dir.glob("*.json"), reverse=True):
        try:
            with open(fp, "r") as f:
                d = json.load(f)
            entries.append({
                "week_end": fp.stem,
                "week_start": d.get("week_start", ""),
                "paper_count": d.get("paper_count", 0),
            })
        except Exception:
            entries.append({"week_end": fp.stem, "week_start": "", "paper_count": 0})

    path = Path("data/archive_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"weeks": entries}, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Archive index: {len(entries)} weeks → {path}")

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Weekly rPPG paper fetcher")
    parser.add_argument("--days", type=int, default=7, help="Look back N days (default: 7)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    now = datetime.utcnow()
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end = now
        start = end - timedelta(days=args.days)

    print(f"═══ rPPG Weekly Paper Fetcher ═══")
    print(f"  Range: {start.date()} → {end.date()}")

    raw = fetch_papers_range(start, end)
    scored = score_and_filter(raw)

    weekly = save_weekly(scored, start, end)
    save_archive(weekly, end)
    build_recent()
    build_archive_index()

    print(f"═══ Done. {len(scored)} relevant papers. ═══")

if __name__ == "__main__":
    main()