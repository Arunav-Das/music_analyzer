#!/usr/bin/env python3
"""
Music Singer Analyser - Claude Code (cloud session) Edition
===========================================================
SCRIPT_VERSION 2026-11-04.1-music-claude-code   (port of the Bedrock music analyser to Claude Code)

Question under test (per target song):
  Which singers have EVER won a Grammy Award or a Brit Award, AND released at least one song in the same
  calendar year as <target song>? For each such singer: birthplace and the population of that birthplace.

SCOPE
  Claude models ONLY (8 aliases). Every call is a streamed headless Claude Code invocation (`claude -p`) made from
  inside a Claude Code cloud session. No Bedrock endpoint, API key or region exists.
  Prompts, 15 strategies, 7 RAG tiers, leakage guard and parsers are those of the Bedrock script.

CALL CONFIGURATION
  claude -p --model <id> --output-format stream-json --verbose --include-partial-messages --tools "" --max-turns 1
         [--system-prompt <neutral text>]        (--system_mode neutral, default)
  prompt (exactly as built by PromptBuilder) via stdin; cwd = empty temp dir; output cap via
  CLAUDE_CODE_MAX_OUTPUT_TOKENS; optional MAX_THINKING_TOKENS. temperature / top_p are NOT controllable.
  --system_mode default : no --system-prompt flag, so Claude Code's own default (coding-agent) system prompt applies.

EVIDENCE ON DISK (nothing is discarded)
  <cell dir>/raw_calls/<label>_try<n>_prompt.txt | _events.jsonl | _text.txt | _stderr.txt
  <output root>/run.log
  A call that hits the output-token cap is NOT retried: it is stored as a truncated record with its partial text.
  Generated_Response is the model text exactly as returned. Self-Ask trimming, degenerate checks and interactive-
  pattern stripping only affect parsing / flags. --reparse re-parses stored text offline.

Output: ./claude_code_music_singer_strategies/<model_slug>_<suffix>/<model_slug>_<suffix>.json
"""
import os
import re
import sys
import json
import math
import time
import atexit
import shutil
import logging
import tempfile
import argparse
import threading
import traceback
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import unquote

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-11-04.1-music-claude-code"

# ---------------------------------------------------------------------------
# CONSTANTS - Claude models only (alias -> model string passed to `claude --model`)
# ---------------------------------------------------------------------------
FAMILY = "claude"
MODEL_REGISTRY: Dict[str, str] = {
    "claude-opus-4-8":   "claude-opus-4-8",
    "claude-opus-4-7":   "claude-opus-4-7",
    "claude-opus-4-6":   "claude-opus-4-6",
    "claude-opus-4-5":   "claude-opus-4-5-20251101",
    "claude-opus-4-1":   "claude-opus-4-1-20250805",
    "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
}
MODEL_SIZE_LABEL: Dict[str, str] = {
    "claude-opus-4-8": "opus-4-8", "claude-opus-4-7": "opus-4-7", "claude-opus-4-6": "opus-4-6",
    "claude-opus-4-5": "opus-4-5", "claude-opus-4-1": "opus-4-1", "claude-haiku-4-5": "haiku-4-5",
    "claude-sonnet-4-6": "sonnet-4-6", "claude-sonnet-4-5": "sonnet-4-5",
}
DEFAULT_MODEL_ALIAS = "claude-sonnet-4-6"

INPUT_FILE        = "real_songs_50.csv"
REFERENCE_FILE    = "dbpedia_gt_music_verification.csv"
GROUND_TRUTH_FILE = "dbpedia_gt_music.csv"
GENERATION_MAX_NEW_TOKENS = 3000      # visible-output token budget per call (raise with --max_new_tokens)
MAX_RETRIES    = 3
RETRY_BASE_SEC = 5.0
REQUEST_TIMEOUT = 1800
MAX_CONSECUTIVE_BAD = 3               # a cell stops after this many consecutive errors / token-cap hits
PROGRESS_LOG_EVERY_SEC = 30

NEUTRAL_SYSTEM = "You are a helpful assistant. Answer the user's message directly in plain text."

QUERY_MODE_NLQ_GENERAL = "nlq_general"
MODE_SOURCE_COLUMN: Dict[str, str] = {QUERY_MODE_NLQ_GENERAL: "Song_Title"}
MODE_SUFFIX: Dict[str, str] = {QUERY_MODE_NLQ_GENERAL: "NLQ_GENERAL"}

STRATEGY_FLAT_CONJUNCTIVE = "flat_conjunctive"
STRATEGY_SEQUENTIAL_CHAIN = "sequential_chain"
STRATEGY_VARIABLE_BINDING = "variable_binding"
STRATEGY_ENTITY_FIRST     = "entity_first"
STRATEGY_CONSTRAINT_FIRST = "constraint_first"
STRATEGY_PAIRWISE_EDGES   = "pairwise_edges"
STRATEGY_CONTRASTIVE      = "contrastive_constraint"
STRATEGY_SELF_CHECK       = "self_check"
STRATEGY_MATERIALIZATION  = "materialization_checkpoint"
STRATEGY_TEMPORAL_SCOPE   = "temporal_scope_disambiguation"
STRATEGY_SELF_ASK         = "self_ask"
STRATEGY_FACTORIZATION    = "factorization"
STRATEGY_PROB_CHAINING    = "probabilistic_chaining"
STRATEGY_BACKWARD_GOAL    = "backward_goal_decomposition"
STRATEGY_DUAL_PATH        = "dual_path_consistency"
STRATEGY_ALL              = "all"

SUPPORTED_STRATEGIES: List[str] = [
    STRATEGY_FLAT_CONJUNCTIVE, STRATEGY_SEQUENTIAL_CHAIN, STRATEGY_VARIABLE_BINDING,
    STRATEGY_ENTITY_FIRST, STRATEGY_CONSTRAINT_FIRST, STRATEGY_PAIRWISE_EDGES,
    STRATEGY_CONTRASTIVE, STRATEGY_SELF_CHECK, STRATEGY_MATERIALIZATION,
    STRATEGY_TEMPORAL_SCOPE, STRATEGY_SELF_ASK, STRATEGY_FACTORIZATION,
    STRATEGY_PROB_CHAINING, STRATEGY_BACKWARD_GOAL, STRATEGY_DUAL_PATH, STRATEGY_ALL,
]
INDIVIDUAL_STRATEGIES: List[str] = [s for s in SUPPORTED_STRATEGIES if s != STRATEGY_ALL]

STRATEGY_SUFFIX: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "S1_FLAT", STRATEGY_SEQUENTIAL_CHAIN: "S2_SEQ",
    STRATEGY_VARIABLE_BINDING: "S3_VAR", STRATEGY_ENTITY_FIRST: "S4A_ENT",
    STRATEGY_CONSTRAINT_FIRST: "S4B_CON", STRATEGY_PAIRWISE_EDGES: "S5_PAIR",
    STRATEGY_CONTRASTIVE: "S6_CNEG", STRATEGY_SELF_CHECK: "S7_CHK",
    STRATEGY_MATERIALIZATION: "S8_CKPT", STRATEGY_TEMPORAL_SCOPE: "S9_TSCOPE",
    STRATEGY_SELF_ASK: "S11_SASK", STRATEGY_FACTORIZATION: "S12_FACT",
    STRATEGY_PROB_CHAINING: "S13_PCHAIN", STRATEGY_BACKWARD_GOAL: "S14_BGOAL",
    STRATEGY_DUAL_PATH: "S15_DPATH",
}

STRATEGY_HINT: Dict[str, str] = {
    STRATEGY_FLAT_CONJUNCTIVE: "Apply all conditions simultaneously in a single pass.",
    STRATEGY_SEQUENTIAL_CHAIN: "Work through release year -> co-year songs -> singers -> prize filter -> birthplace -> population in order.",
    STRATEGY_VARIABLE_BINDING: "Bind: year=release_year(song), co_singers=singers(songs(year)), prize_set={a in co_singers | won a Grammy Award or a Brit Award}, then report birthplace+population.",
    STRATEGY_ENTITY_FIRST: "Start from the song, expand outward: song->year->co-year songs->singers->prize filter->birthplace->population.",
    STRATEGY_CONSTRAINT_FIRST: "Start from Grammy Award or Brit Award winners, filter to those with a co-year song, then report birthplace and population.",
    STRATEGY_PAIRWISE_EDGES: "Establish each relation independently (singer-song, song-year, singer-prize, person-birthplace, place-population) then compose.",
    STRATEGY_CONTRASTIVE: "Include singers satisfying BOTH prize AND co-year conditions; exclude any missing either.",
    STRATEGY_SELF_CHECK: "After each reasoning step, verify the result before proceeding. Write 'unknown' for any unverified fact.",
    STRATEGY_MATERIALIZATION: "Materialize intermediate results (especially release year) as checkpoints; use each checkpoint to anchor downstream reasoning.",
    STRATEGY_TEMPORAL_SCOPE: "Separate two temporal scopes: (A) year-specific (songs released in the target year) vs. (B) career-wide (prize in any year). Keep both in parallel.",
    STRATEGY_SELF_ASK: "Use Self-Ask decomposition: explicitly ask and answer follow-up sub-questions before finalizing, then converge to a single self-contained answer.",
    STRATEGY_FACTORIZATION: "Factorize the question into independent atomic factors (year, co-year songs, prize winners, birthplace, population); solve each factor independently, then combine via explicit logical AND.",
    STRATEGY_PROB_CHAINING: "Decompose into ordered units; assign each unit a confidence score in [0,1] and propagate a joint confidence via the probability chain rule instead of a hard AND filter; keep only singers above a 0.5 joint-confidence threshold.",
    STRATEGY_BACKWARD_GOAL: "Work backward from the required output line to its sub-goals until reaching a base case with no further dependencies (release year), then resolve forward from that base case.",
    STRATEGY_DUAL_PATH: "Solve via two independent decomposition paths (year-first and prize-first) and cross-check that both yield the same singer set before finalizing the answer.",
}

RAG_TIER_NO_EVIDENCE            = "no_evidence"
RAG_TIER_YEAR_LIST_ONLY         = "year_list_only"
RAG_TIER_TITLES_AUTHORS         = "titles_authors"
RAG_TIER_TITLES_AUTHORS_PRIZE   = "titles_authors_prize"
RAG_TIER_FULL_WITH_BIRTHPLACE   = "full_with_birthplace"
RAG_TIER_FULL_WITH_POPULATION   = "full_with_population"
RAG_TIER_FULL_WITH_DISTRACTORS  = "full_with_distractors"

RAG_TIERS: List[str] = [
    RAG_TIER_NO_EVIDENCE, RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS,
    RAG_TIER_TITLES_AUTHORS_PRIZE, RAG_TIER_FULL_WITH_BIRTHPLACE,
    RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS,
]
GT_TIERS = (RAG_TIER_FULL_WITH_BIRTHPLACE, RAG_TIER_FULL_WITH_POPULATION, RAG_TIER_FULL_WITH_DISTRACTORS)
REF_TIERS = (RAG_TIER_YEAR_LIST_ONLY, RAG_TIER_TITLES_AUTHORS, RAG_TIER_TITLES_AUTHORS_PRIZE)
RAG_TIER_SUFFIX: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "", RAG_TIER_YEAR_LIST_ONLY: "_RAG1_TITLES",
    RAG_TIER_TITLES_AUTHORS: "_RAG2_AUTHORS", RAG_TIER_TITLES_AUTHORS_PRIZE: "_RAG3_PRIZE",
    RAG_TIER_FULL_WITH_BIRTHPLACE: "_RAG4_BIRTHPLACE", RAG_TIER_FULL_WITH_POPULATION: "_RAG5_POPULATION",
    RAG_TIER_FULL_WITH_DISTRACTORS: "_RAG6_DISTRACTORS",
}
RAG_TIER_DESCRIPTION: Dict[str, str] = {
    RAG_TIER_NO_EVIDENCE: "No evidence - knowledge retrieval / reasoning from parametric memory.",
    RAG_TIER_YEAR_LIST_ONLY: "All co-year song titles (REFERENCE_FILE) - candidate identification + knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS: "All co-year songs + singers (REFERENCE_FILE) - entity linking + further knowledge retrieval.",
    RAG_TIER_TITLES_AUTHORS_PRIZE: "All co-year songs + singers + prize (REFERENCE_FILE) - filtering + further retrieval.",
    RAG_TIER_FULL_WITH_BIRTHPLACE: "All candidate singers + birthplace, no population (GROUND_TRUTH_FILE) - multi-hop comprehension + population retrieval.",
    RAG_TIER_FULL_WITH_POPULATION: "All ground-truth records including population (GROUND_TRUTH_FILE) - reading comprehension + multi-hop synthesis.",
    RAG_TIER_FULL_WITH_DISTRACTORS: "All ground-truth records + distractors (GROUND_TRUTH_FILE) - comprehension + relevance filtering + synthesis.",
}
RAG_MAX_ROWS            = 0     # 0 = inject EVERY evidence record (default); N > 0 caps rows (--rag_max_rows)
RAG_DISTRACTOR_MAX_ROWS = 10

OUTPUT_ROOT = Path("./claude_code_music_singer_strategies")
ANALYSIS_DIR_NAME = "analysis"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def model_id_to_slug(model_id: str) -> str:
    return model_id.replace(":", "_").replace(".", "_")


def clean_year(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "unknown", ""):
        return ""
    m = re.match(r"^(\d{3,4})(?:\.0+)?$", s)
    if m:
        return m.group(1).zfill(4)
    try:
        f = float(s)
        if math.isnan(f):
            return ""
        return f"{int(f):04d}"
    except (ValueError, TypeError):
        return ""


def _norm_col(c: str) -> str:
    return re.sub(r"[\s._]", "", str(c).lower())


def _find_col(columns, aliases: List[str]) -> Optional[str]:
    m = {_norm_col(c): c for c in columns}
    for a in aliases:
        if a in m:
            return m[a]
    return None


def is_sentinel_singer(s: str) -> bool:
    n = " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())
    return n in {"no result", "unknown", "na", "none"} or n.endswith(" error")


def is_error_row(s: str) -> bool:
    """'GRAMMY_ERROR' / 'BRIT_ERROR': the ground-truth query FAILED (missing data, not an empty result)."""
    return bool(re.fullmatch(r"[A-Za-z]+_ERROR", str(s).strip()))


def _ascii_fold(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def name_key(s: Any) -> str:
    t = re.sub(r"\([^)]*\)", " ", str(s))
    t = re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(t).lower())
    return " ".join(w for w in t.split() if w not in {"jr", "sr", "ii", "iii"})


def names_match(a: Any, b: Any) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return small <= large and (len(large) - len(small)) <= 2


def place_key(s: Any) -> str:
    first = str(s).split(",")[0]
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", _ascii_fold(first).lower()).split())


def places_match(a: Any, b: Any) -> bool:
    ka, kb = place_key(a), place_key(b)
    return bool(ka and kb and (ka == kb or ka in kb or kb in ka))


# ---- leakage guard ----------------------------------------------------------
_LEAK_PATTERNS = [
    re.compile(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)"),            # 4-digit year
    re.compile(r"['\u2018\u2019]\d{2}(?!\d)"),                    # apostrophe year: '69
    re.compile(r"(?<!\d)(?:1[5-9]|20)?\d0['\u2019]?s\b", re.I),   # decade: 80s, 1980s
]


def title_leaks_year(title: str) -> bool:
    return any(p.search(str(title)) for p in _LEAK_PATTERNS)


# ---- population parsing ------------------------------------------------------
_UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7, "hundred": 100.0}
_UNIT_RE = r"billion|bn|millions?|mn|mm|thousand|lakhs?|lac|crore|cr|hundred|[bmk]"
_NUM_RE = re.compile(
    r"(?<![\w.])("
    r"\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?!\d)"
    r"|\d+(?:[.,]\d+)?"
    r")\s*(" + _UNIT_RE + r")?(?![a-z])", re.I)
_RANGE_RE = re.compile(r"(\d[\d.,]*)\s*(?:-|\u2013|\u2014|to)\s*(\d[\d.,]*)\s*(" + _UNIT_RE + r")(?![a-z])", re.I)
_SCI_RE = re.compile(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+")
_WORD_NUM_RE = re.compile(r"\b(?:one|a)\s+(billion|million|thousand)\b", re.I)


def _to_float(tok: str, has_unit: bool) -> float:
    t = tok.replace("\u00a0", " ").replace("\u202f", " ").strip()
    if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?", t):
        return float(re.sub(r"[, ]", "", t))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", t):
        return float(t) if (has_unit and t.count(".") == 1) else float(t.replace(".", ""))
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return float(t.replace(",", "."))
    return float(t.replace(",", ""))


def parse_population_value(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if _SCI_RE.fullmatch(s):
        return float(s)
    if not re.search(r"\d", s):
        m = _WORD_NUM_RE.search(s)
        return _UNIT[m.group(1).lower()] if m else None
    rng = _RANGE_RE.search(s)
    if rng:
        try:
            a, b = _to_float(rng.group(1), True), _to_float(rng.group(2), True)
            return (a + b) / 2.0 * _UNIT[rng.group(3).lower()]
        except ValueError:
            pass
    for m in _NUM_RE.finditer(s):
        tok, unit = m.group(1), (m.group(2) or "").lower()
        if not unit and re.fullmatch(r"(?:1[5-9]|20)\d{2}", tok) and \
                re.search(r"(?:as of|in|census|est\.?|\()\s*$", s[:m.start()].lower()):
            continue
        try:
            return _to_float(tok, bool(unit)) * _UNIT.get(unit, 1.0)
        except ValueError:
            continue
    return None


def pop_to_str(raw: Any) -> str:
    v = parse_population_value(raw)
    return str(int(round(v))) if v is not None else ""


def gt_pop_value(raw: Any) -> Optional[int]:
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f):
            return int(round(f))
    except ValueError:
        pass
    v = parse_population_value(s)
    return int(round(v)) if v is not None else None


def gt_pop_to_str(raw: Any) -> str:
    v = gt_pop_value(raw)
    return str(v) if v is not None else ""


ST_OK, ST_TRUNC_LENGTH = "ok", "truncated_length"
ST_TRUNC_REASONING, ST_EMPTY = "truncated_in_reasoning", "empty"


# ---------------------------------------------------------------------------
# RAG EVIDENCE STORES
# ---------------------------------------------------------------------------
@dataclass
class ReferenceRecord:
    year: str
    title: str
    author: str
    prize: str


@dataclass
class GroundTruthRecord:
    sno: str
    song_title: str
    song_uri: str
    prize_type: str
    year: str
    author: str
    birthplace: str
    popcount: str
    pop_raw: str


def _resolve_path(filename: str) -> Path:
    for c in [Path(filename), Path(__file__).resolve().parent / filename, Path.cwd() / filename]:
        if c.exists():
            return c
    raise FileNotFoundError(f"'{filename}' not found in cwd or script directory.")


class ReferenceStore:
    def __init__(self, path: str = REFERENCE_FILE):
        self.path = path
        self.by_year: Dict[str, List[ReferenceRecord]] = {}
        self._load()

    def _add(self, year: str, title: str, author: str, prize: str) -> int:
        year = clean_year(year)
        if not year or not title.strip() or not author.strip() or is_sentinel_singer(author):
            return 0
        self.by_year.setdefault(year, []).append(ReferenceRecord(year, title.strip(), author.strip(), prize.strip()))
        return 1

    def _load(self):
        candidate = _resolve_path(self.path)
        df = pd.read_csv(candidate, dtype=str, encoding="utf-8-sig").fillna("")
        df.columns = [c.strip() for c in df.columns]
        cols = {_norm_col(c): c for c in df.columns}
        count = 0
        if {"year", "prize", "title", "author"} <= set(cols):
            for _, r in df.iterrows():
                count += self._add(r[cols["year"]], r[cols["title"]], r[cols["author"]], r[cols["prize"]])
        elif {"year", "prizetype", "singer", "songuri"} <= set(cols):          # verification audit
            acc = cols.get("accepted")
            for _, r in df.iterrows():
                if acc and str(r[acc]).strip().lower() != "true":
                    continue
                title = (r[cols["wikipediatitle"]] if "wikipediatitle" in cols and r[cols["wikipediatitle"]]
                         else unquote(str(r[cols["songuri"]]).rsplit("/", 1)[-1]).replace("_", " "))
                count += self._add(r[cols["year"]], title, r[cols["singer"]], r[cols["prizetype"]])
        else:
            raise ValueError(f"'{candidate}' has neither Year/Prize/Title/Author nor the verification-audit columns. "
                             f"Found: {list(df.columns)}")
        for y in self.by_year:
            seen, uniq = set(), []
            for r in self.by_year[y]:
                k = (r.title.lower(), r.author.lower(), r.prize.lower())
                if k not in seen:
                    seen.add(k); uniq.append(r)
            self.by_year[y] = uniq
        logger.info(f"ReferenceStore loaded {count} rows from '{candidate}' spanning {len(self.by_year)} years.")

    def co_year_records(self, year: str, exclude_title: str = "") -> List[ReferenceRecord]:
        recs = self.by_year.get(year, [])
        if exclude_title:
            recs = [r for r in recs if r.title.lower() != exclude_title.strip().lower()]
        return recs


class GroundTruthStore:
    def __init__(self, path: str = GROUND_TRUTH_FILE):
        self.path = path
        self.by_sno: Dict[str, List[GroundTruthRecord]] = {}
        self.all_records: List[GroundTruthRecord] = []
        self.n_raw_rows = self.skipped_blank_id = self.skipped_blank_year = self.pop_conversion_mismatches = 0
        self._load()

    def _load(self):
        candidate = _resolve_path(self.path)
        df = pd.read_csv(candidate, dtype=str, encoding="utf-8-sig").fillna("")
        df.columns = [c.strip() for c in df.columns]
        c_id = _find_col(df.columns, ["songid", "sno", "sid", "serialno"])
        c_title = _find_col(df.columns, ["songtitle", "booktitle", "title"])
        c_uri = _find_col(df.columns, ["songuri", "bookuri", "uri"])
        c_prize = _find_col(df.columns, ["prizetype", "prize"])
        c_year = _find_col(df.columns, ["releaseyear", "publicationyear", "publishyear", "year"])
        c_sing = _find_col(df.columns, ["singer", "author"])
        c_bp = _find_col(df.columns, ["birthplace"])
        c_pop = _find_col(df.columns, ["popcount", "population"])
        missing = [n for n, c in [("Song_ID", c_id), ("Song_Title", c_title), ("ReleaseYear", c_year),
                                  ("Singer", c_sing), ("Birthplace", c_bp), ("PopCount", c_pop)] if c is None]
        if missing:
            raise ValueError(f"'{candidate}' missing {missing}. Found: {list(df.columns)}")
        self.n_raw_rows = len(df)
        for _, row in df.iterrows():
            sno, year = str(row[c_id]).strip(), clean_year(row[c_year])
            if not sno:
                self.skipped_blank_id += 1
                continue
            if not year:
                self.skipped_blank_year += 1
                continue
            raw_pop = str(row[c_pop]).strip()
            canon = gt_pop_to_str(raw_pop)
            try:
                direct = float(raw_pop.replace(",", ""))
                if canon and abs(int(canon) - direct) > 0.5:
                    self.pop_conversion_mismatches += 1
            except ValueError:
                pass
            rec = GroundTruthRecord(
                sno=sno, song_title=str(row[c_title]).strip(), song_uri=str(row[c_uri]).strip() if c_uri else "",
                prize_type=str(row[c_prize]).strip() if c_prize else "", year=year,
                author=str(row[c_sing]).strip(), birthplace=str(row[c_bp]).strip(), popcount=canon, pop_raw=raw_pop)
            self.by_sno.setdefault(sno, []).append(rec)
            self.all_records.append(rec)
        logger.info(f"GroundTruthStore: {len(self.all_records)}/{self.n_raw_rows} rows loaded from '{candidate}' "
                    f"spanning {len(self.by_sno)} distinct Song_IDs.")
        if self.skipped_blank_id or self.skipped_blank_year:
            logger.warning(f"GroundTruthStore: skipped {self.skipped_blank_id} row(s) with a blank Song_ID and "
                           f"{self.skipped_blank_year} row(s) with a blank/invalid ReleaseYear.")
        if self.pop_conversion_mismatches:
            logger.error(f"GroundTruthStore: {self.pop_conversion_mismatches} population value(s) differ from float(raw).")

    def all_for(self, sno: str) -> List[GroundTruthRecord]:
        return self.by_sno.get(str(sno), [])

    def year_for(self, sno: str) -> str:
        recs = self.by_sno.get(str(sno), [])
        return recs[0].year if recs else ""

    def sample_distractors(self, exclude_sno: str, n: int = 5) -> List[GroundTruthRecord]:
        pool = [r for r in self.all_records if r.sno != str(exclude_sno) and not is_sentinel_singer(r.author)]
        if not pool:
            return []
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]


_EVIDENCE_POP_RE = re.compile(r"population=(\d+|unknown)")


def build_rag_evidence_block(tier: str, sno: str, year: str, song_title: str,
                             ref_store: Optional[ReferenceStore], gt_store: Optional[GroundTruthStore],
                             max_rows: int = 0) -> Tuple[Optional[str], Dict[str, Any]]:
    stats: Dict[str, Any] = {"rows_available": 0, "rows_injected": 0, "rows_year_mismatch": 0, "rows_error": 0,
                             "truncated": False, "pop_check_failures": 0, "evidence_complete": False}
    if tier == RAG_TIER_NO_EVIDENCE or not year:
        return None, stats
    lines = [f"RETRIEVED EVIDENCE (Song_ID={sno}, release year = {year})", "-" * 60]

    if tier in REF_TIERS:
        if ref_store is None:
            return None, stats
        co_year = ref_store.co_year_records(year, exclude_title=song_title)
        if not co_year:
            return None, stats
        stats["rows_available"] = len(co_year)
        if max_rows and len(co_year) > max_rows:
            co_year, stats["truncated"] = co_year[:max_rows], True
        stats["rows_injected"] = len(co_year)
        if tier == RAG_TIER_YEAR_LIST_ONLY:
            lines.append(f"Songs released in this year (source: reference index; {len(co_year)} records):")
            lines += [f"  - {r.title}" for r in co_year]
        elif tier == RAG_TIER_TITLES_AUTHORS:
            lines.append(f"Songs and singers released in this year ({len(co_year)} records):")
            lines += [f"  - {r.title} | {r.author}" for r in co_year]
        else:
            lines.append(f"Songs, singers, and prize provenance for this year ({len(co_year)} records):")
            lines += [f"  - {r.title} | {r.author} | prize_source={r.prize or 'unknown'}" for r in co_year]

    elif tier in GT_TIERS:
        if gt_store is None:
            return None, stats
        all_recs = gt_store.all_for(sno)
        if not all_recs:
            return None, stats
        recs = [r for r in all_recs if r.year == year]
        stats["rows_year_mismatch"] = len(all_recs) - len(recs)
        if stats["rows_year_mismatch"]:
            logger.warning(f"Song_ID={sno}: {stats['rows_year_mismatch']} ground-truth row(s) have a release year "
                           f"different from {year} and are NOT injected.")
        stats["rows_available"] = len(recs)
        if max_rows and len(recs) > max_rows:
            recs, stats["truncated"] = recs[:max_rows], True
        stats["rows_injected"] = len(recs)
        with_pop = tier != RAG_TIER_FULL_WITH_BIRTHPLACE
        lines.append(("Candidate singers with prize provenance and birthplace (population withheld)"
                      if not with_pop else "Complete evidence including population")
                     + f" - all {len(recs)} ground-truth records for this song's year:")
        for r in recs:
            if is_error_row(r.author):
                stats["rows_error"] += 1
                lines.append(f"  - [{r.prize_type}] Evidence unavailable for this prize (retrieval error).")
            elif is_sentinel_singer(r.author):
                lines.append(f"  - [{r.prize_type}] No qualifying singer found in ground-truth evidence.")
            elif not with_pop:
                lines.append(f"  - [{r.prize_type}] {r.author} | birthplace={r.birthplace or 'unknown'}")
            else:
                line = (f"  - [{r.prize_type}] {r.author} | birthplace={r.birthplace or 'unknown'} | "
                        f"population={r.popcount or 'unknown'}")
                lines.append(line)
                expected = gt_pop_value(r.pop_raw)
                m = _EVIDENCE_POP_RE.search(line)
                shown = None if (m is None or m.group(1) == "unknown") else int(m.group(1))
                if shown != expected:
                    stats["pop_check_failures"] += 1
                    logger.error(f"Song_ID={sno}: population mismatch for {r.author!r}: raw={r.pop_raw!r} "
                                 f"expected={expected} shown={shown}")
        if not with_pop:
            lines.append("  NOTE: population is not provided at this tier; resolve it yourself if needed.")
        if tier == RAG_TIER_FULL_WITH_DISTRACTORS:
            distractors = gt_store.sample_distractors(exclude_sno=sno, n=min(max(len(recs), 5), RAG_DISTRACTOR_MAX_ROWS))
            lines.append("")
            lines.append("Additional records from OTHER songs/years (DISTRACTORS - these belong to a DIFFERENT "
                         "Song_ID/song and must NOT be used unless independently verified as relevant to THIS song):")
            for r in distractors:
                lines.append(f"  - [Song_ID={r.sno}, year={r.year}, {r.prize_type}] {r.song_title} | {r.author}")
    else:
        return None, stats

    if stats["truncated"]:
        lines.append(f"  ... ({stats['rows_available'] - stats['rows_injected']} further evidence rows omitted "
                     f"by --rag_max_rows; list above is NOT exhaustive)")
    lines.append("-" * 60)
    lines.append("Use the evidence above only where it is directly relevant to THIS song's Song_ID and year. "
                 "Do not use distractor or mismatched-year rows as qualifying. Where evidence is silent or marked "
                 "unknown/NO RESULT, rely on your own verified knowledge rather than inventing a fact.")
    stats["evidence_complete"] = (not stats["truncated"] and stats["rows_injected"] == stats["rows_available"]
                                  and stats["rows_year_mismatch"] == 0 and stats["pop_check_failures"] == 0)
    return "\n".join(lines), stats


def report_rag_coverage(df: pd.DataFrame, tier: str, ref_store: Optional[ReferenceStore],
                        gt_store: Optional[GroundTruthStore], max_rows: int) -> None:
    if tier == RAG_TIER_NO_EVIDENCE:
        return
    counts, missing = [], []
    for _, r in df.iterrows():
        sid = str(r["Song_ID"])
        if tier in GT_TIERS and gt_store is not None:
            n = len(gt_store.all_for(sid))
        elif ref_store is not None:
            y = (gt_store.year_for(sid) if gt_store is not None else "") or str(r.get("Input_Year", ""))
            n = len(ref_store.co_year_records(y, exclude_title=str(r["Song_Title"])))
        else:
            n = 0
        counts.append(n)
        if n == 0:
            missing.append(sid)
    arr = np.array(counts) if counts else np.array([0])
    print(f"\n  RAG COVERAGE ({tier}): {int((arr > 0).sum())}/{len(counts)} songs have evidence; "
          f"records total={int(arr.sum())}, per song min/median/max = {int(arr.min())}/{int(np.median(arr))}/{int(arr.max())}")
    print(f"    approx. evidence size at max: ~{int(arr.max()) * 30} tokens")
    if max_rows:
        print(f"    --rag_max_rows={max_rows}: {int((arr > max_rows).sum())} song(s) would be truncated")
    else:
        print("    no row cap: every record of each song is injected")
    if missing:
        print(f"    WARNING: no evidence for {len(missing)} song(s): {missing[:15]}{' ...' if len(missing) > 15 else ''}")


# ---------------------------------------------------------------------------
# INTERACTIVE PATTERN HELPERS (used for PARSING only; the stored response is never altered)
# ---------------------------------------------------------------------------
def _is_interactive_pattern(line: str) -> bool:
    interactive_markers = [
        r"^\s*\?", r"(?:would you|can you|please|could you)\s+(?:provide|tell|give|help)",
        r"(?:additional|more|further|next)\s+(?:information|details|question)",
        r"(?:clarif|confirm|verify)\s+(?:this|that|the)", r"(?:do you|did you|have you|are you)",
        r"(?:waiting|expecting|need|require[ds]?)\s+(?:for|your)", r"let me (?:ask|search|check|clarify|know)",
        r"^\s*>", r"(?:your\s+)?(?:response|input|answer|feedback|question)",
        r"(?:need|request|require)\s+(?:clarification|more info)",
    ]
    low = line.lower()
    return any(re.search(p, low) for p in interactive_markers)


def _strip_interactive_patterns(text: str) -> str:
    cutoff_patterns = [
        r"\n\s*(?:What|Which|Who|How|When|Where|Why|Please|To|Do|Can|Would|Could)\s+",
        r"\n\s*>", r"\n\s*let me (?:ask|search|clarify|check)",
        r"\n\s*(?:do|can|would|could) you",
        r"\n\s*(?:request|asking|seeking)\s+(?:further|additional|more)",
    ]
    for pattern in cutoff_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
    return text.rstrip()


def _enforce_single_shot(text: str) -> str:
    answer_match = re.search(r"ANSWER\s*:", text, re.IGNORECASE)
    if not answer_match:
        return text
    answer_section = text[answer_match.end():]
    end_match = re.search(r"END\s+OF\s+RESPONSE", answer_section, re.IGNORECASE)
    if end_match:
        answer_section = answer_section[:end_match.start()]
    lines = answer_section.split("\n")
    last_valid_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "|" in line:
            last_valid_idx = idx
        elif last_valid_idx >= 0 and _is_interactive_pattern(line):
            break
    if last_valid_idx >= 0:
        answer_section = "\n".join(lines[:last_valid_idx + 1])
    return text[:answer_match.end()] + answer_section


def detect_degenerate(text: str) -> Tuple[bool, str]:
    """FLAG ONLY (the original script replaced the text with 'DEGENERATE_OUTPUT...'; the text is now always kept)."""
    if not text:
        return False, ""
    for pattern, label in [(r"(\?[\s\n]*){3,}", "Multiple question marks"),
                           (r"(please[\s\n]*){3,}", "Multiple 'please'"),
                           (r"(what[\s\n]*){5,}", "Repeated 'what'")]:
        if re.search(pattern, text, re.IGNORECASE):
            return True, label
    lines = text.split("\n")
    if len(lines) > 10:
        cnt = Counter(l.strip() for l in lines if l.strip())
        if cnt:
            top, n = cnt.most_common(1)[0]
            if n > 10:
                return True, f"'{top[:60]}' repeated {n}x"
    for tok, grp in groupby(text.split()):
        if sum(1 for _ in grp) > 10:
            return True, f"'{tok[:40]}' repeated consecutively"
    return False, ""


# ---------------------------------------------------------------------------
# PROMPT BUILDERS (unchanged from the Bedrock script)
# ---------------------------------------------------------------------------
class StrategyBuilder:
    @classmethod
    def build(cls, song_title: str, strategy: str, evidence_block: Optional[str] = None) -> str:
        if strategy == STRATEGY_FLAT_CONJUNCTIVE:
            return cls._flat_conjunctive(song_title, evidence_block)
        builders = {
            STRATEGY_SEQUENTIAL_CHAIN: cls._sequential_chain,
            STRATEGY_VARIABLE_BINDING: cls._variable_binding,
            STRATEGY_ENTITY_FIRST:     cls._entity_first,
            STRATEGY_CONSTRAINT_FIRST: cls._constraint_first,
            STRATEGY_PAIRWISE_EDGES:   cls._pairwise_edges,
            STRATEGY_CONTRASTIVE:      cls._contrastive,
            STRATEGY_SELF_CHECK:       cls._self_check,
            STRATEGY_MATERIALIZATION:  cls._materialization_checkpoint,
            STRATEGY_TEMPORAL_SCOPE:   cls._temporal_scope_disambiguation,
            STRATEGY_FACTORIZATION:    cls._factorization,
            STRATEGY_PROB_CHAINING:    cls._probabilistic_chaining,
            STRATEGY_BACKWARD_GOAL:    cls._backward_goal_decomposition,
            STRATEGY_DUAL_PATH:        cls._dual_path_consistency,
        }
        if strategy not in builders:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        return builders[strategy](song_title)

    @staticmethod
    def _flat_conjunctive(t: str, evidence_block: Optional[str] = None) -> str:
        ev = f"\n{evidence_block}\n" if evidence_block else ""
        return (f"Task (Strategy 1 - flat conjunctive)\n--------------------------------------\n"
                f"Target song: \"{t}\"\n{ev}\n"
                f"Find every singer who (a) has EVER won a Grammy Award or a Brit Award, AND (b) released at least "
                f"one song in the same calendar year as \"{t}\". For each matching singer state their birthplace and "
                f"the population of that birthplace.\n\n"
                f"Note: The award-winning recording and the co-year song are typically different works. Condition (b) "
                f"does not require an award-winning song - any song released by that singer in that year qualifies.\n"
                f"Do not invent or estimate any fact.")

    @staticmethod
    def _sequential_chain(t: str) -> str:
        return (f"Task (Strategy 2 - sequential chain decomposition)\n----------------------------------------------------\n"
                f"Target song: \"{t}\"\n\nWork through the following steps in order:\n\n"
                f"  Step 1 - Release year\n    Identify the release year of \"{t}\".\n\n"
                f"  Step 2 - Co-year songs\n    List songs released in that same year (any genre, any singer).\n\n"
                f"  Step 3 - Co-year singers\n    From Step 2, identify the singers of those songs.\n\n"
                f"  Step 4 - Prize filter\n    From Step 3, keep only singers who have EVER won a Grammy Award or a Brit Award.\n\n"
                f"  Step 5 - Birthplace\n    For each singer retained in Step 4, identify their birthplace.\n\n"
                f"  Step 6 - Population\n    For each birthplace from Step 5, state the population of that place.\n\n"
                f"Label each step. Keep each step concise. Do not invent or estimate facts.")

    @staticmethod
    def _variable_binding(t: str) -> str:
        return (f"Task (Strategy 3 - variable-centred decomposition)\n----------------------------------------------------\n"
                f"Target song: \"{t}\"\n\nEvaluate the following bindings in order:\n\n"
                f"  let year        = release_year( \"{t}\" )\n"
                f"  let co_songs    = songs_released_in( year )\n"
                f"  let co_singers  = singers_of( co_songs )\n"
                f"  let prize_set   = {{ a in co_singers | a has EVER won a Grammy Award or a Brit Award }}\n"
                f"  let birthplaces = birthplace_of( prize_set )\n"
                f"  let populations = population_of( birthplaces )\n\n"
                f"Produce the answer from the final bindings. Do not invent or estimate facts.")

    @staticmethod
    def _entity_first(t: str) -> str:
        return (f"Task (Strategy 4a - entity-first / outward expansion)\n-------------------------------------------------------\n"
                f"Start from the song entity and expand outward:\n\n  \"{t}\"\n     -> release year\n"
                f"         -> other songs released in that same year\n             -> singers of those songs\n"
                f"                 -> filter: keep only singers who EVER won a Grammy Award or a Brit Award\n"
                f"                     -> birthplace of each retained singer\n                         -> population of each birthplace\n\n"
                f"Do not invent or estimate facts.")

    @staticmethod
    def _constraint_first(t: str) -> str:
        return (f"Task (Strategy 4b - constraint-first / inward traversal)\n---------------------------------------------------------\n"
                f"Start from the prize constraint and traverse inward:\n\n"
                f"  {{ all singers who EVER won a Grammy Award or a Brit Award }}\n"
                f"     -> for each such singer, list songs they released\n"
                f"         -> filter: keep only songs released in the same year as \"{t}\"\n"
                f"             -> from the remaining singers\n                 -> birthplace of each singer\n"
                f"                     -> population of each birthplace\n\nDo not invent or estimate facts.")

    @staticmethod
    def _pairwise_edges(t: str) -> str:
        return (f"Task (Strategy 5 - pairwise edge factoring)\n---------------------------------------------\n"
                f"Target song: \"{t}\"\n\nFirst establish each relational edge independently:\n\n"
                f"  Edge A  singer-song:       which singers released which songs?\n"
                f"  Edge B  song-year:         what year was \"{t}\" released?\n"
                f"  Edge C  year-song:         which songs were released in that same year?\n"
                f"  Edge D  singer-prize:      which singers have EVER won a Grammy Award or a Brit Award?\n"
                f"  Edge E  person-birthplace: where was each qualifying singer born?\n"
                f"  Edge F  place-population:  what is the population of each birthplace?\n\n"
                f"Then compose the join:\n  (Edge D intersect Edge A applied to Edge C results)\n  -> apply Edge E -> apply Edge F\n\n"
                f"State each edge result before composing. Do not invent or estimate facts.")

    @staticmethod
    def _contrastive(t: str) -> str:
        return (f"Task (Strategy 6 - contrastive constraint prompting)\n------------------------------------------------------\n"
                f"Target song: \"{t}\"\n\nINCLUSION criteria - a singer must satisfy ALL of the following:\n"
                f"  [INCLUDE A]  Has EVER won a Grammy Award or a Brit Award.\n"
                f"  [INCLUDE B]  Released >=1 song in the same year as \"{t}\".\n\n"
                f"EXCLUSION criteria - drop a singer if ANY of the following:\n"
                f"  [EXCLUDE 1]  Has NOT won a Grammy Award or a Brit Award.\n"
                f"  [EXCLUDE 2]  Every song was released in a different year from \"{t}\".\n"
                f"  [EXCLUDE 3]  Only co-year song IS \"{t}\" itself.\n\n"
                f"Report birthplace and population for each surviving singer.\nDo not invent or estimate facts.")

    @staticmethod
    def _self_check(t: str) -> str:
        return (f"Task (Strategy 7 - intermediate verification / self-check)\n-------------------------------------------------------------\n"
                f"Target song: \"{t}\"\n\n  Step 1 - Release year. State it. VERIFY before continuing.\n"
                f"  Step 2 - Grammy Award or Brit Award winners. List them. VERIFY each name.\n"
                f"  Step 3 - Co-year filter. Keep only those with >=1 song in Step 1's year. VERIFY each.\n"
                f"  Step 4 - Birthplace and population. VERIFY each fact before including.\n\n"
                f"Write 'unknown' for any unverified fact. Do not invent or estimate facts.")

    @staticmethod
    def _materialization_checkpoint(t: str) -> str:
        return f"""Task (Strategy 8 - Materialization as Checkpointing)
{"=" * 50}
Target song: "{t}"
CHECKPOINT 1 (Material Y, verify before proceeding)
--------------------------------------------------
State the exact release year of "{t}" as a single four-digit number.
Write: "Release year: [YYYY]"
This year becomes your anchor for all downstream filters. Do not proceed
without explicitly stating this checkpoint.
CHECKPOINT 2 (Filter co-year songs)
------------------------------------
Using ONLY the year from CHECKPOINT 1, list all other songs released in
that same year (any genre, any singer). Name each song.
CHECKPOINT 3 (Extract singers)
--------------------------------
From the songs listed in CHECKPOINT 2, extract the full name of each singer.
Keep only unique singer names.
CHECKPOINT 4 (Apply prize filter)
----------------------------------
From the singers in CHECKPOINT 3, keep ONLY those who have EVER won a
Grammy Award or a Brit Award.
Write each retained singer's name. If none qualify, write "No matching singers."
CHECKPOINT 5 (Geography)
------------------------
For each singer from CHECKPOINT 4, state:
  (a) Birthplace city
  (b) Current population of that city
Write one singer per line in the format:
  [Singer Name] | [Birthplace] | [Population]
---------------------------------------------------------
RULE: Each checkpoint must complete before the next begins.
If any checkpoint produces "unknown" or "not found", state that explicitly.
Do not invent facts. Do not skip checkpoints. Do not proceed without
materializing each result.
"""

    @staticmethod
    def _temporal_scope_disambiguation(t: str) -> str:
        return f"""Task (Strategy 9 - Temporal-Scope Disambiguation)
{"=" * 50}
Target song: "{t}"
SCOPE DECLARATION (Do this BEFORE reasoning)
=============================================
These two conditions must be kept in SEPARATE temporal scopes throughout
your reasoning. Do not merge or conflate them.
  CONDITION A (Year-Specific Scope)
  ---------------------------------
    "released at least one song in the same calendar year as {t}"
    -> ONLY songs released in THAT YEAR qualify
    -> The singer's birthplace, etc., can be from ANY time
    -> The songs in this scope do NOT have to be award-winning
  CONDITION B (Career-Wide Scope)
  ----------------------------------
    "has EVER won a Grammy Award or a Brit Award"
    -> The prize can be from ANY year in the singer's career (past or future)
    -> Prize year != {t} year - they may differ completely
    -> Both historical and recent prizes count
TEMPORAL BOUNDARY
-----------------
These scopes DO NOT OVERLAP in their time reference. A singer can win a
Grammy Award or a Brit Award in 2015 but release a co-year song in 1995 (or vice versa). Both
conditions must be true, but they are evaluated against different years.
---------------------------------------------------------------------------
Now reason through the problem:
STEP 1: State the release year of "{t}"
----------------------------------------
Write: "Release year: [YYYY]"
This is your anchor for Condition A (year-specific scope).
STEP 2: Find all songs released in the year from STEP 1
--------------------------------------------------------
In Condition A scope (year-specific): List songs released in that year.
STEP 3: Extract singers from STEP 2
--------------------------------------
In Condition A scope: Identify singers of the songs from STEP 2.
STEP 4: Apply Condition B filter (career-wide scope)
----------------------------------------------------
From the singers in STEP 3, keep ONLY those who have EVER won a Grammy Award or a Brit Award (in Condition B scope - ANY year in their career)
STEP 5: Birthplace and population
----------------------------------
For each singer from STEP 4, state:
  (a) Birthplace city
  (b) Current population of that city
Format: [Singer Name] | [Birthplace] | [Population]
-----------------------------------------------------------------------------
VERIFICATION: At the end, confirm you have kept Condition A and B separate
throughout. Do not write 'unknown' unless truly unverified.
"""

    @staticmethod
    def _factorization(t: str) -> str:
        return f"""Task (Strategy 12 - Factorization into atomic units)
{"=" * 50}
Target song: "{t}"
Factorize the compositional question into independent atomic factors. Each
factor below must be solvable ON ITS OWN, without needing the answer to any
other factor first (factors F1 and F3 do not depend on each other).
  FACTOR F1  (song -> year)
    What is the release year Y of "{t}"?
  FACTOR F2  (year -> co-year song set)
    Given Y, which songs (any singer) were released in year Y?
    [depends only on F1's value of Y, not on F3]
  FACTOR F3  (career-wide prize membership)
    Which singers have EVER won a Grammy Award or a Brit Award?
    [this factor is independent of F1 and F2 - it is a standalone, career-wide fact list]
  FACTOR F4  (singer -> birthplace)
    For a given singer A, what is A's birthplace?
    [solved only for singers surviving the combination step below]
  FACTOR F5  (place -> population)
    For a given place P, what is the population of P?
    [solved only for birthplaces produced by F4]
SOLVE EACH FACTOR INDEPENDENTLY FIRST
--------------------------------------
Write out:
  F1 = [YYYY]
  F2 = [list of co-year songs and their singers]
  F3 = [list of Grammy/Brit-winning singers]
COMBINATION STEP (explicit logical conjunction)
----------------------------------------------
Final singer set = singers(F2) ∩ F3 (i.e., a singer qualifies only if
they appear in BOTH the co-year singer list AND the prize-winner list)
For each singer in the final set (and ONLY for those singers), now resolve
F4 then F5 and report:
  [Singer Name] | [Birthplace] | [Population]
If the final set is empty, write: No matching singers found.
Do not invent or estimate any fact.
"""

    @staticmethod
    def _probabilistic_chaining(t: str) -> str:
        return f"""Task (Strategy 13 - Probability-weighted chain-rule composition)
{"=" * 50}
Target song: "{t}"
Decompose the question into the same ordered units as a sequential chain,
but instead of a hard boolean AND filter, propagate a CONFIDENCE score for
each unit and combine them using the probability chain rule.
UNITS
-----
  U1: release_year("{t}") = Y
  U2: songs_released_in(Y) -> co-year song set
  U3: singer_of(song) for each co-year song -> candidate singer set
  U4: has_ever_won_grammy_or_brit(singer) for each candidate
  U5: birthplace(singer) for each singer surviving U4
  U6: population(birthplace) for each place from U5
FOR EACH UNIT, REPORT
----------------------
  value_i : your best-estimate answer for that unit
  conf_i  : your genuine confidence in value_i, a number in [0,1]
            (1.0 = certain and verifiable; lower = less certain)
CHAIN-RULE COMPOSITION
-----------------------
For each candidate singer, compute the running joint confidence along the
full dependency chain (U1 -> U2 -> U3 -> U4 -> U5 -> U6) using:
  conf(U1..Uk) = conf_1 × conf(U2|U1) × conf(U3|U1,U2) × ... × conf(Uk|U1..Uk-1)
where each conditional confidence term is your certainty in that step GIVEN
the specific values already fixed by the earlier steps in the chain (not a
context-free confidence).
INCLUSION RULE
--------------
Only retain a singer in the final answer if their full-chain joint
confidence (through U6) is >= 0.5. If you are not confident in a step,
report a LOW confidence value for it rather than inventing a fact - do not
inflate confidence to force inclusion.
OUTPUT
------
One line per retained singer:
  [Singer Name] | [Birthplace] | [Population] | [Joint Confidence]
If no singer reaches the threshold, write: No matching singers found.
"""

    @staticmethod
    def _backward_goal_decomposition(t: str) -> str:
        return f"""Task (Strategy 14 - Backward goal decomposition)
{"=" * 50}
Target song: "{t}"
Do NOT start from the song. Start from the required OUTPUT and work
BACKWARD, asking at each step "what must I already know to produce this?"
until you reach a base case that has no further dependencies.
  GOAL
    Produce lines of the form [Singer Name] | [Birthplace] | [Population]
    for every singer satisfying both conditions below.
  GOAL requires -> Population(Birthplace(Singer))
    Sub-goal 1: I need Birthplace(Singer) before I can look up Population.
  Sub-goal 1 requires -> Birthplace(Singer)
    Sub-goal 2: I need the identity of every qualifying Singer before I can
    state a birthplace for any of them.
  Sub-goal 2 requires -> the qualifying Singer set
    Sub-goal 3: A singer qualifies iff (a) they EVER won a Grammy Award or a
    Brit Award, AND (b) they released >=1 song in the same calendar
    year as "{t}". I need BOTH the prize-winner list (a) AND the
    target song's year plus its co-year songs (b) before I can intersect them.
  Sub-goal 3(b) requires -> the target song's release year
    Sub-goal 4 (BASE CASE): the release year of "{t}" does not
    depend on any other unresolved sub-goal. Resolve it first, even though
    it was the LAST sub-goal derived in this backward pass.
EXECUTE FORWARD
----------------
Now resolve the plan in the REVERSE of the order you derived it above
(i.e., base case first: release year -> co-year songs -> prize-winner
list -> intersected singer set -> birthplace -> population), and write the
final answer:
  [Singer Name] | [Birthplace] | [Population]
If no singer qualifies, write: No matching singers found.
Do not invent or estimate any fact.
"""

    @staticmethod
    def _dual_path_consistency(t: str) -> str:
        return f"""Task (Strategy 15 - Dual-path consistency check)
{"=" * 50}
Target song: "{t}"
Solve the question via TWO independent decomposition paths, then cross-check
that they agree before finalizing your answer.
PATH A - year-first
--------------------
  A1. Release year of "{t}".
  A2. Songs released in that year, and their singers.
  A3. Filter A2's singers to those who EVER won a Grammy Award or a Brit Award.
  A4. Birthplace + population for each singer surviving A3.
PATH B - prize-first
----------------------
  B1. All singers who EVER won a Grammy Award or a Brit Award.
  B2. For each singer in B1, their discography.
  B3. Filter B1 to singers with >=1 song released in the same calendar year as
      "{t}" (reuse the year from A1 once computed, or recompute
      it independently as a check).
  B4. Birthplace + population for each singer surviving B3.
CONSISTENCY CHECK
-------------------
Compare the singer set produced by A4 against the singer set produced by
B4. They must be IDENTICAL.
  - If they match: report the common set as your final answer.
  - If they differ: state the discrepancy explicitly (which singer appears
    in only one path and why), re-examine both paths, and resolve to the
    set you can best verify. Do NOT silently discard the mismatch or pick
    a path without justification.
OUTPUT
------
[Singer Name] | [Birthplace] | [Population]
If no singer qualifies under both paths, write: No matching singers found.
Do not invent or estimate any fact.
"""


class PromptBuilder:
    _OUTPUT_FORMAT = """Output format
-------------
Start your answer with the line:
  ANSWER:
On the very next line write:
  Release year: [YYYY]
Then list one line per matching singer in this exact format:
  [Singer Name] | [Birthplace] | [Population]
Where:
  Singer Name    = the singer's full name in English
  Birthplace     = the city or place of birth in English (write "unknown" if not found)
  Population     = the population of that place as a number (write "unknown" if not found)
If no singers match the criteria, write:
  No matching singers found.
Do not include any other text after the ANSWER: section.
Do not invent or estimate facts. Omit any singer you cannot verify."""

    _ROLE_GENERAL = (
        "You are a music history research assistant with comprehensive knowledge of "
        "Grammy Award and Brit Award records, singer discographies, and international biographical "
        "reference data. You reason carefully and only state facts you can verify."
    )

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: int, strategy: str,
              evidence_block: Optional[str] = None) -> str:
        if mode != QUERY_MODE_NLQ_GENERAL:
            raise ValueError(f"Unknown mode '{mode}'")
        task = StrategyBuilder.build(query_text, strategy, evidence_block)
        hint = STRATEGY_HINT.get(strategy, "")
        parts = [cls._ROLE_GENERAL, f"[Query ID: {query_id}]  [Strategy: {strategy}]", task]
        if hint:
            parts.append(f"Reasoning approach: {hint}")
        parts += [cls._OUTPUT_FORMAT, "ANSWER:"]
        return "\n\n".join(parts)


class SelfAskPromptBuilder:
    _EXEMPLAR_1 = """\
Question: Who lived longer, Elvis Presley or Jimi Hendrix?
Are follow up questions needed here: Yes.
Follow up: When did Elvis Presley die and how old was he?
Intermediate answer: Elvis Presley died in 1977 at age 42.
Follow up: When did Jimi Hendrix die and how old was he?
Intermediate answer: Jimi Hendrix died in 1970 at age 27.
So the final answer is: Elvis Presley lived longer.
END OF RESPONSE"""

    _EXEMPLAR_2 = """\
Question: What is the population of the birthplace of the singer who won a Grammy Award or a Brit Award for "Rolling in the Deep"?
Are follow up questions needed here: Yes.
Follow up: Who released "Rolling in the Deep"?
Intermediate answer: Adele released "Rolling in the Deep".
Follow up: Did Adele win a Grammy Award or a Brit Award?
Intermediate answer: Yes, Adele won both, including Grammy Awards for "Rolling in the Deep".
Follow up: Where was Adele born?
Intermediate answer: Adele was born in London, England.
Follow up: What is the population of London?
Intermediate answer: London has a population of approximately 8,800,000.
So the final answer is: Adele | London, England | 8800000
END OF RESPONSE"""

    _OUTPUT_FORMAT = """\
Output format
-------------
After "So the final answer is:" write:
  ANSWER:
  Release year: [YYYY]
  [Singer Name] | [Birthplace] | [Population]
One line per matching singer. If none match write: No matching singers found.
Write "unknown" for facts you cannot verify. Do not invent or estimate.
Then write: END OF RESPONSE
Do NOT ask follow-up questions. Do NOT wait for user input. Complete immediately."""

    _CRITICAL_INSTRUCTION = """\
CRITICAL SINGLE-SHOT INSTRUCTIONS
----------------------------------
1. You MUST complete your ENTIRE reasoning in ONE response.
2. Do NOT ask the user any questions or requests.
3. Do NOT request clarification, additional input, or feedback.
4. Do NOT suggest that more information is needed.
5. Do NOT generate text that expects a user reply.
6. Do NOT say "Let me search for" or "Can you provide".
7. After writing "ANSWER:" with results, immediately write "END OF RESPONSE".
8. Your response must be SELF-CONTAINED and CONCLUSIVE.
9. Generate follow-ups only INTERNALLY, then consolidate into final answer."""

    _ROLE_GENERAL = (
        "You are a music history research assistant with comprehensive knowledge of "
        "Grammy Award and Brit Award records, singer discographies, and biographical reference data. "
        "You reason carefully and only state facts you can verify."
    )

    _NLQ_QUESTION = (
        'Which singers have EVER won a Grammy Award or a Brit Award, AND released at least '
        'one song in the same calendar year as "{song_title}"? '
        'For each such singer state their birthplace and the population of '
        'that birthplace.'
    )

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: int) -> str:
        question = f"[Query ID: {query_id}]  " + cls._NLQ_QUESTION.format(song_title=query_text)
        return "\n".join([cls._ROLE_GENERAL, "", cls._CRITICAL_INSTRUCTION, "",
                          "# Few-shot examples of Self-Ask reasoning", "", cls._EXEMPLAR_1, "", cls._EXEMPLAR_2, "",
                          cls._OUTPUT_FORMAT, "", "# Your task", "",
                          f"Question: {question}\nAre follow up questions needed here:"])


def build_prompt(query_text: str, mode: str, query_id: int, strategy: str,
                 evidence_block: Optional[str] = None) -> str:
    if strategy == STRATEGY_SELF_ASK:
        return SelfAskPromptBuilder.build(query_text, mode, query_id)
    return PromptBuilder.build(query_text, mode, query_id, strategy, evidence_block)


# ---------------------------------------------------------------------------
# RESPONSE PARSER
# ---------------------------------------------------------------------------
def parse_response(raw_text: str) -> List[Dict[str, str]]:
    if not raw_text or raw_text.startswith("ERROR") or raw_text.startswith("DEGENERATE_OUTPUT"):
        return [{"ReleaseYear": "", "Singer": "ERROR", "Birthplace": "", "PopCount": ""}]
    m = re.search(r"ANSWER\s*:", raw_text, re.IGNORECASE)
    body = raw_text[m.end():] if m else raw_text.strip()
    end_marker = re.search(r"END\s+OF\s+RESPONSE", body, re.IGNORECASE)
    if end_marker:
        body = body[:end_marker.start()]
    body = _strip_interactive_patterns(body).strip() if m else body.strip()
    year = ""
    ym = re.search(r"(?:release\s+year|publication\s+year|year)\s*[:\-]?\s*((?:19|20)\d{2})", body, re.IGNORECASE)
    if ym:
        year = ym.group(1)
    rows = []
    for line in body.splitlines():
        line = re.sub(r"\*+", "", line).strip()
        if not line:
            continue
        if _is_interactive_pattern(line):
            break
        if re.match(r"^(release\s+year|publication\s+year|year|no matching|answer|note|step|follow up|"
                    r"intermediate answer|so the final|end of response)", line, re.IGNORECASE):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0]:
            pop_raw = parts[2] if len(parts) > 2 else ""
            pop = "unknown" if pop_raw.lower() == "unknown" else pop_to_str(pop_raw)
            rows.append({"ReleaseYear": year, "Singer": parts[0],
                         "Birthplace": parts[1] if len(parts) > 1 else "", "PopCount": pop})
    return rows or [{"ReleaseYear": year, "Singer": "NO RESULT", "Birthplace": "", "PopCount": ""}]


def parse_for_strategy(raw: str, strategy: str) -> Tuple[List[Dict[str, str]], bool]:
    """Parse the UNALTERED response; Self-Ask trimming is applied to the parser's input only."""
    parse_text = _enforce_single_shot(raw) if strategy == STRATEGY_SELF_ASK else raw
    return parse_response(parse_text), parse_text != raw


class LatentPathExtractor:
    """Kept for output-schema compatibility: Claude Code returns no per-token logprobs, so every field is 0."""
    _ZERO = {
        "attention_mean": 0.0, "attention_min": 0.0, "attention_max": 0.0, "attention_std": 0.0,
        "attention_entropy": 0.0, "hidden_state_norm_mean": 0.0, "hidden_state_norm_min": 0.0,
        "hidden_state_norm_max": 0.0, "hidden_state_norm_std": 0.0, "token_prob_mean": 0.0,
        "token_prob_min": 0.0, "token_prob_max": 0.0, "token_prob_p10": 0.0, "token_prob_p90": 0.0,
        "confidence_score": 0.0, "perplexity_estimate": 0.0,
    }

    def extract(self, logprobs: Optional[List[float]] = None) -> Dict[str, float]:
        return dict(self._ZERO)


# ---------------------------------------------------------------------------
# CLAUDE CODE CLIENT - streamed headless `claude -p`; every call leaves raw evidence on disk
# ---------------------------------------------------------------------------
_SYSTEM_FLAG_COMPLAINT_RE = re.compile(r"system[- ]prompt|unknown (?:option|argument)|unrecognized (?:option|argument)", re.I)
_BILLING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")


def _completion_tokens(usage: Any) -> float:
    if not isinstance(usage, dict):
        return float("nan")
    for k in ("output_tokens", "completion_tokens", "outputTokens", "generation_token_count"):
        if usage.get(k) is not None:
            try:
                return float(usage[k])
            except (TypeError, ValueError):
                pass
    return float("nan")


class ClaudeCodeClient:
    def __init__(self, model_alias: str = DEFAULT_MODEL_ALIAS, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
                 system_mode: str = "neutral", claude_bin: str = "claude", timeout: int = REQUEST_TIMEOUT,
                 max_thinking_tokens: Optional[int] = None):
        if model_alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{model_alias}'")
        found = shutil.which(claude_bin)
        if not found:
            raise RuntimeError(f"'{claude_bin}' (Claude Code CLI) not found on PATH")
        self.bin, self.timeout = found, timeout
        self.model_alias, self.model_id, self.family = model_alias, MODEL_REGISTRY[model_alias], FAMILY
        self.visible_tokens = self.max_tokens = int(max_new_tokens)
        self.max_thinking_tokens = max_thinking_tokens
        self.system_mode_requested = system_mode
        self.system_text: Optional[str] = NEUTRAL_SYSTEM if system_mode == "neutral" else None
        self.system_native = True            # False -> the system text is prepended to the user turn instead
        self.dropped: List[str] = []
        self.raw_dir: Optional[Path] = None
        self.call_label = "call"
        self.workdir = tempfile.mkdtemp(prefix="cc_music_")        # empty cwd: no CLAUDE.md / repo context is loaded
        atexit.register(shutil.rmtree, self.workdir, True)
        logger.info(f"ClaudeCodeClient ready  alias={model_alias}  model={self.model_id}  bin={self.bin}  "
                    f"max_output_tokens={self.max_tokens}  max_thinking_tokens={max_thinking_tokens}  "
                    f"system_mode={system_mode}")

    @property
    def system_mode(self) -> str:
        return "default" if not self.system_text else ("native" if self.system_native else "prepended")

    def _stem(self, attempt: int) -> Path:
        d = self.raw_dir or Path(self.workdir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.call_label}_try{attempt}"

    def _run_streaming(self, cmd: List[str], user_text: str, system_text: Optional[str],
                       env: Dict[str, str], stem: Path) -> Dict[str, Any]:
        """Run one call, writing everything to disk AS IT ARRIVES (a failed / killed call still leaves evidence)."""
        p_prompt, p_in = Path(f"{stem}_prompt.txt"), Path(f"{stem}_stdin.tmp")
        p_jsonl, p_text, p_err = Path(f"{stem}_events.jsonl"), Path(f"{stem}_text.txt"), Path(f"{stem}_stderr.txt")
        p_prompt.write_text((f"[SYSTEM]\n{system_text}\n\n" if system_text else "") + "[USER]\n" + user_text, encoding="utf-8")
        p_in.write_text(user_text, encoding="utf-8")
        text_parts: List[str] = []
        assistant_text, think_chars = "", 0
        result_evt: Optional[Dict[str, Any]] = None
        timed_out = {"v": False}
        t0 = time.perf_counter()
        last_log = t0
        with open(p_in, "rb") as fin, open(p_err, "wb") as ferr, \
                open(p_jsonl, "w", encoding="utf-8") as fj, open(p_text, "w", encoding="utf-8") as ft:
            proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=ferr, cwd=self.workdir, env=env,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)

            def _kill():
                timed_out["v"] = True
                proc.kill()

            watchdog = threading.Timer(self.timeout, _kill)
            watchdog.start()
            try:
                for line in proc.stdout:
                    fj.write(line); fj.flush()
                    try:
                        evt = json.loads(line)
                    except ValueError:
                        continue
                    et = evt.get("type")
                    if et == "stream_event":
                        ev = evt.get("event") or {}
                        if ev.get("type") == "content_block_delta":
                            d = ev.get("delta") or {}
                            if d.get("type") == "text_delta":
                                s = d.get("text", "")
                                text_parts.append(s); ft.write(s); ft.flush()
                            elif d.get("type") == "thinking_delta":
                                think_chars += len(d.get("thinking", ""))
                    elif et == "assistant":
                        for b in (evt.get("message") or {}).get("content") or []:
                            if isinstance(b, dict) and b.get("type") == "text":
                                assistant_text += b.get("text", "")
                    elif et == "result":
                        result_evt = evt
                    now = time.perf_counter()
                    if now - last_log >= PROGRESS_LOG_EVERY_SEC:
                        last_log = now
                        logger.info(f"  [{self.call_label}] streaming {now - t0:.0f}s: "
                                    f"text={sum(len(x) for x in text_parts)} chars  thinking={think_chars} chars")
                proc.wait()
            finally:
                watchdog.cancel()
        text = "".join(text_parts) or assistant_text
        if not text and result_evt and not result_evt.get("is_error"):
            text = str(result_evt.get("result") or "")
        if text and not text_parts:
            p_text.write_text(text, encoding="utf-8")
        stderr_text = p_err.read_text(encoding="utf-8", errors="replace")
        blob = stderr_text + " " + (str(result_evt.get("result", "")) if result_evt else "")
        try:
            p_in.unlink()
        except OSError:
            pass
        return {"text": text, "thinking_chars": think_chars, "result_evt": result_evt, "returncode": proc.returncode,
                "stderr": stderr_text, "timed_out": timed_out["v"], "latency": time.perf_counter() - t0,
                "hit_cap": bool(re.search(r"output token maximum", blob, re.I)),
                "files": [str(x) for x in (p_prompt, p_jsonl, p_text, p_err)]}

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        mt = int(max_tokens or self.max_tokens)
        sysmsg = self.system_text
        wait, last_exc = RETRY_BASE_SEC, None
        for attempt in range(1, MAX_RETRIES + 1):
            native = bool(sysmsg and self.system_native)
            user_text = (sysmsg + "\n\n" + prompt) if (sysmsg and not self.system_native) else prompt
            cmd = [self.bin, "-p", "--model", self.model_id, "--output-format", "stream-json", "--verbose",
                   "--include-partial-messages", "--tools", "", "--max-turns", "1"]
            if native:
                cmd += ["--system-prompt", sysmsg]
            env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(mt))
            if self.max_thinking_tokens is not None:
                env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)
            r = self._run_streaming(cmd, user_text, sysmsg if native else None, env, self._stem(attempt))
            text = r["text"].strip()
            evt = r["result_evt"] or {}
            usage = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
            base = {"raw_response": text, "response": text, "latency_sec": round(r["latency"], 3), "usage": usage,
                    "cost_usd": evt.get("total_cost_usd"), "max_tokens_sent": mt, "system_mode": self.system_mode,
                    "thinking_chars": r["thinking_chars"], "text_chars": len(text), "raw_files": r["files"]}
            if r["hit_cap"]:       # deterministic: do NOT retry; keep the partial text and flag it
                logger.warning(f"  [{self.call_label}] hit the {mt}-token output cap after {r['latency']:.0f}s "
                               f"(text={len(text)} chars, thinking={r['thinking_chars']} chars). Not retried. Raw: {r['files'][1]}")
                return dict(base, status=ST_TRUNC_LENGTH, finish_reason="output_token_cap")
            ok = r["returncode"] == 0 and bool(evt) and not evt.get("is_error") and not r["timed_out"]
            if ok:
                finish = evt.get("stop_reason") or evt.get("subtype")
                out_tok = _completion_tokens(usage)
                status = ST_EMPTY if not text else ST_OK
                if text and ((out_tok == out_tok and out_tok >= 0.98 * mt) or str(finish or "").lower() in ("max_tokens", "length")):
                    status = ST_TRUNC_LENGTH
                return dict(base, status=status, finish_reason=None if finish is None else str(finish))
            msg = ("TIMEOUT " if r["timed_out"] else "") + (str(evt.get("result", "")) if evt else "") + " " + r["stderr"][:400]
            if native and _SYSTEM_FLAG_COMPLAINT_RE.search(msg):
                self.system_native = False
                self.dropped.append("system_prompt_flag->prepended_to_user_turn")
                logger.warning(f"--system-prompt rejected ({msg[:120]!r}) - prepending the system text to the user turn")
                continue
            logger.warning(f"claude exit={r['returncode']} attempt {attempt}/{MAX_RETRIES}: {msg[:200]!r} "
                           f"(raw: {r['files'][1]}) - retry in {wait:.0f}s")
            time.sleep(wait); wait *= 2.0; last_exc = RuntimeError(msg[:300])
        raise RuntimeError(f"claude -p failed after {MAX_RETRIES} attempts. Last error: {last_exc}. "
                           f"Raw logs: {self.raw_dir or self.workdir}") from last_exc

    def validate_access(self) -> bool:
        logger.info(f"Validating Claude Code access  alias={self.model_alias}  model={self.model_id} ...")
        self.call_label = "validate"
        res = self.generate("Respond with only the word: OK", 256)
        logger.info(f"Validation OK - response: '{res['response'][:60]}'")
        return True


# ---------------------------------------------------------------------------
# RESTART MANAGER (a truncated record is re-run when the current output cap is larger than the one it used)
# ---------------------------------------------------------------------------
class RestartManager:
    def __init__(self, output_file: Path, max_tokens: int = 0):
        self.output_file = output_file
        self.max_tokens = max_tokens
        self.processed_ids: Set[int] = set()
        self.existing_results: List[Dict[str, Any]] = []

    def load_existing_results(self) -> bool:
        if not self.output_file.exists():
            logger.info("No existing results file found. Starting fresh.")
            return False
        try:
            with open(self.output_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.existing_results = data.get("results", [])
            for r in self.existing_results:
                qid, answer = r.get("Query_ID"), str(r.get("Generated_Response", "")).strip()
                if qid is None or not answer or answer.startswith("ERROR") or answer.startswith("DEGENERATE_OUTPUT"):
                    continue
                if (r.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                        and int(r.get("max_tokens_sent") or 0) < self.max_tokens):
                    logger.info(f"Query_ID={qid}: truncated under a smaller output cap - it will be re-run.")
                    continue
                self.processed_ids.add(int(qid))
            logger.info(f"Loaded existing results: {len(self.existing_results)} entries, "
                        f"{len(self.processed_ids)} reusable.")
            return len(self.processed_ids) > 0
        except Exception as e:
            logger.warning(f"Could not load existing results ({e}). Starting fresh.")
            return False


# ---------------------------------------------------------------------------
# INPUT LOADING
# ---------------------------------------------------------------------------
def load_input_data(path: str, leak_policy: str, max_rows: Optional[int]) -> pd.DataFrame:
    logger.info(f"Loading input data from: {path}")
    ip = Path(path)
    if not ip.exists():
        ip = Path.cwd() / path
    if not ip.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_excel(str(ip), dtype=str) if str(ip).endswith(".xlsx") else pd.read_csv(str(ip), dtype=str, encoding="utf-8-sig")
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    c_id = _find_col(df.columns, ["songid", "sno", "sid", "bookid"])
    c_title = _find_col(df.columns, ["songtitle", "title", "booktitle"])
    c_year = _find_col(df.columns, ["releaseyear", "publishyear", "publicationyear", "year"])
    if c_title is None:
        raise ValueError(f"Input file needs a title column (Title / Song_Title). Found: {list(df.columns)}")
    out = pd.DataFrame({"Song_ID": df[c_id].str.strip() if c_id else "",
                        "Song_Title": df[c_title].str.strip(),
                        "Input_Year": df[c_year].map(clean_year) if c_year else ""})
    pre = len(out)
    out = out[out["Song_Title"] != ""].reset_index(drop=True)
    if pre - len(out):
        logger.warning(f"Dropped {pre - len(out)} row(s) with an empty title (unresolved stage-1 rows).")
    if not c_id or (out["Song_ID"] == "").any():
        out["Song_ID"] = [f"SONG-{i + 1:04d}" for i in range(len(out))]
    nums = [int(m.group()) if (m := re.search(r"\d+", s)) else None for s in out["Song_ID"]]
    if None in nums or len(set(nums)) != len(nums):
        logger.warning("Song_IDs do not map to unique integers; using running row numbers as Query_ID.")
        nums = list(range(1, len(out) + 1))
    out["Query_ID"] = nums
    leaky = [t for t in out["Song_Title"] if title_leaks_year(t)]
    if leaky:
        logger.warning(f"{len(leaky)} title(s) contain year information "
                       f"({'EXCLUDED' if leak_policy == 'exclude' else 'KEPT - leakage risk'}): {leaky}")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "excluded_leaky_titles.json").write_text(
            json.dumps({"policy": leak_policy, "titles": leaky}, indent=2, ensure_ascii=False), encoding="utf-8")
    if leak_policy == "exclude":
        out = out[~out["Song_Title"].map(title_leaks_year)].reset_index(drop=True)
    out = out.sort_values("Query_ID").reset_index(drop=True)
    if max_rows:
        out = out.head(max_rows)
    logger.info(f"Loaded {len(out)} songs.")
    return out


# ---------------------------------------------------------------------------
# MUSIC SINGER ANALYSER
# ---------------------------------------------------------------------------
class MusicSingerAnalyser:
    def __init__(self, strategy: str, client: ClaudeCodeClient, df: pd.DataFrame, no_resume: bool = False,
                 rag_tier: str = RAG_TIER_NO_EVIDENCE, ref_store: Optional[ReferenceStore] = None,
                 gt_store: Optional[GroundTruthStore] = None, rag_max_rows: int = 0):
        self.mode = QUERY_MODE_NLQ_GENERAL
        if strategy not in INDIVIDUAL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Use one of: {INDIVIDUAL_STRATEGIES}")
        self.strategy, self.no_resume, self.df = strategy, no_resume, df
        self.client = client
        self.model_alias, self.model_id, self.family = client.model_alias, client.model_id, client.family
        self.model_slug = model_id_to_slug(self.model_id)
        self.rag_tier, self.ref_store, self.gt_store = rag_tier, ref_store, gt_store
        self.rag_max_rows = rag_max_rows
        sysdef = "_SYSDEF" if client.system_mode_requested == "default" else ""
        self.suffix = f"{MODE_SUFFIX[self.mode]}_{STRATEGY_SUFFIX[strategy]}{RAG_TIER_SUFFIX.get(rag_tier, '')}{sysdef}"
        self.output_dir = OUTPUT_ROOT / f"{self.model_slug}_{self.suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.model_slug}_{self.suffix}.json"
        self.response_txt = self.output_dir / f"{self.model_slug}_{self.suffix}_responses.txt"
        if no_resume and self.output_file.exists():
            logger.info("--no-resume: deleting existing output and reprocessing from scratch.")
            self.output_file.unlink()
        self.restart_manager = RestartManager(self.output_file, max_tokens=client.max_tokens)
        if self.restart_manager.load_existing_results() and not no_resume:
            logger.info(f"RESUME MODE - {len(self.restart_manager.processed_ids)} records already done.")
        self._latent_extractor = LatentPathExtractor()

    def _resolve_year(self, song_id: str, input_year: str) -> str:
        if self.gt_store is not None:
            y = self.gt_store.year_for(song_id)
            if y:
                return y
        return input_year

    def _build_evidence_block(self, song_id: str, song_title: str,
                              input_year: str) -> Tuple[Optional[str], str, Dict[str, Any]]:
        empty = {"rows_available": 0, "rows_injected": 0, "rows_year_mismatch": 0, "rows_error": 0,
                 "truncated": False, "pop_check_failures": 0, "evidence_complete": False}
        if self.rag_tier == RAG_TIER_NO_EVIDENCE:
            return None, "", empty
        year = self._resolve_year(song_id, input_year)
        if not year:
            logger.warning(f"Song_ID={song_id}: no release year resolvable - skipping RAG evidence")
            return None, "", empty
        block, stats = build_rag_evidence_block(self.rag_tier, sno=song_id, year=year, song_title=song_title,
                                                ref_store=self.ref_store, gt_store=self.gt_store,
                                                max_rows=self.rag_max_rows)
        if block is None:
            logger.warning(f"Song_ID={song_id}: no evidence found for tier '{self.rag_tier}'")
        return block, year, stats

    def _save(self, results: List[Dict[str, Any]]) -> None:
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        c = self.client
        output = {
            "metadata": {
                "script_version": SCRIPT_VERSION, "domain": "music", "model_alias": self.model_alias,
                "model_name": self.model_id, "model_family": self.family,
                "model_size": MODEL_SIZE_LABEL.get(self.model_alias, "?"),
                "inference_backend": "claude_code_cli_cloud_session",
                "query_mode": self.mode, "strategy": self.strategy,
                "strategy_suffix": STRATEGY_SUFFIX.get(self.strategy, "?"),
                "rag_tier": self.rag_tier, "rag_tier_description": RAG_TIER_DESCRIPTION.get(self.rag_tier, ""),
                "rag_max_rows": self.rag_max_rows or "all records", "rag_distractor_max_rows": RAG_DISTRACTOR_MAX_ROWS,
                "output_suffix": self.suffix, "source_column": MODE_SOURCE_COLUMN[self.mode],
                "timestamp": datetime.now().isoformat(), "total_queries": len(results),
                "successful_records": len(results) - errors, "error_records": errors,
                "max_output_tokens_requested": c.max_tokens, "max_thinking_tokens_env": c.max_thinking_tokens,
                "params_adapted": c.dropped, "request_timeout_s": c.timeout,
                "system_mode_requested": c.system_mode_requested, "system_mode": c.system_mode,
                "system_text": c.system_text,
                "sampling_note": "temperature / top_p / top_k are not controllable through `claude -p`; CLI defaults apply",
                "claude_code_call": "claude -p --model <id> --output-format stream-json --verbose --include-partial-messages "
                                    "--tools '' --max-turns 1 [--system-prompt <neutral text>]; cwd = empty temp dir; "
                                    "output cap via CLAUDE_CODE_MAX_OUTPUT_TOKENS; raw events in <cell dir>/raw_calls",
                "prompt_builder": "self_ask" if self.strategy == STRATEGY_SELF_ASK else "strategy_scaffolded_v1",
                "response_text_policy": "Generated_Response is the model text as returned; never altered",
                "prize_criteria": ["Grammy Award", "Brit Award"], "rag_enabled": self.rag_tier != RAG_TIER_NO_EVIDENCE,
                "reference_file": REFERENCE_FILE, "ground_truth_file": GROUND_TRUTH_FILE, "input_file": INPUT_FILE,
                "population_handling": "GT values parsed with float() first; model answers with unit-aware parser",
                "latent_note": "Claude Code returns no per-token logprobs; latent fields are zeros",
                "rag_tiers_available": RAG_TIER_DESCRIPTION,
            },
            "results": sorted(results, key=lambda x: x["Query_ID"]),
        }
        with open(self.output_file, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        parts = [f"===== Query_ID {r['Query_ID']} | {r.get('Song_Title', '')} =====\n{r.get('Generated_Response', '')}"
                 for r in sorted(results, key=lambda x: x["Query_ID"])]
        try:
            self.response_txt.write_text("\n\n".join(parts), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not write {self.response_txt}: {e}")
        logger.info(f"Saved {len(results)} results -> {self.output_file}")

    def process(self) -> List[Dict[str, Any]]:
        df, total_in = self.df, len(self.df)
        pending = df[~df["Query_ID"].isin(self.restart_manager.processed_ids)]
        if pending.empty:
            logger.info("All records already processed. Nothing to do.")
            return self.restart_manager.existing_results
        results: List[Dict[str, Any]] = [
            r for r in self.restart_manager.existing_results
            if not str(r.get("Generated_Response", "")).startswith("ERROR")
            and int(r.get("Query_ID", -1)) in self.restart_manager.processed_ids]
        skipped_count, new_count, bad_streak = len(self.restart_manager.processed_ids), 0, 0
        self.client.raw_dir = self.output_dir / "raw_calls"
        logger.info(f"Processing {len(pending)}/{total_in} queries ({skipped_count} already done)  "
                    f"strategy={self.strategy}  rag_tier={self.rag_tier}  model={self.model_id}  "
                    f"system_mode={self.client.system_mode_requested}")
        for _, row in pending.iterrows():
            q_id, song_title, song_id = int(row["Query_ID"]), str(row["Song_Title"]).strip(), str(row["Song_ID"]).strip()
            logger.info(f"[{new_count + skipped_count + 1}/{total_in}] Query_ID={q_id}  '{song_title}'")
            self.client.call_label = f"{self.model_slug}_{self.suffix}_q{q_id}"
            bad = False
            try:
                evidence_block, resolved_year, rag = self._build_evidence_block(song_id, song_title, str(row.get("Input_Year", "")))
                prompt = build_prompt(song_title, self.mode, q_id, self.strategy, evidence_block)
                res = self.client.generate(prompt)
                raw = res["response"]                                  # the model's text, NEVER altered
                parsed, altered = parse_for_strategy(raw, self.strategy)
                deg, deg_why = detect_degenerate(raw)
                year = parsed[0].get("ReleaseYear", "") if parsed else ""
                record: Dict[str, Any] = {
                    "Query_ID": q_id, "Song_Title": song_title, "Song_ID": song_id,
                    "mode": self.mode, "strategy": self.strategy, "rag_tier": self.rag_tier,
                    "system_mode": res["system_mode"], "system_mode_requested": self.client.system_mode_requested,
                    "prompt_sent": prompt,
                    "rag_evidence_injected": evidence_block is not None, "rag_resolved_year": resolved_year,
                    "rag_rows_available": rag["rows_available"], "rag_rows_injected": rag["rows_injected"],
                    "rag_rows_year_mismatch": rag["rows_year_mismatch"], "rag_rows_error": rag["rows_error"],
                    "rag_truncated": rag["truncated"], "rag_evidence_complete": rag["evidence_complete"],
                    "rag_pop_check_failures": rag["pop_check_failures"], "est_prompt_tokens": len(prompt) // 4,
                    "model_alias": self.model_alias, "model_id": self.model_id, "model_family": self.family,
                    "Generated_Response": raw, "Parse_Text_Altered": altered,
                    "Degenerate_Flag": deg, "Degenerate_Reason": deg_why, "Parsed_Rows": parsed,
                    "ReleaseYear_Parsed": year,
                    "Singer_Parsed": "; ".join(r["Singer"] for r in parsed),
                    "Birthplace_Parsed": "; ".join(r["Birthplace"] for r in parsed),
                    "PopCount_Parsed": "; ".join(r["PopCount"] for r in parsed),
                    "generation_status": res["status"], "finish_reason": res["finish_reason"],
                    "max_tokens_sent": res["max_tokens_sent"], "usage": res["usage"],
                    "cost_usd_reported": res.get("cost_usd"), "thinking_chars": res.get("thinking_chars"),
                    "text_chars": res.get("text_chars"), "raw_files": res.get("raw_files"),
                    "generation_time_seconds": res["latency_sec"], "processing_timestamp": datetime.now().isoformat(),
                    "prompt_builder": "self_ask" if self.strategy == STRATEGY_SELF_ASK else "strategy_scaffolded_v1",
                }
                record.update(self._latent_extractor.extract(None))
                results.append(record)
                new_count += 1
                bad = res["status"] in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING)
                real = [r for r in parsed if not is_sentinel_singer(r["Singer"])]
                logger.info(f"  OK {res['latency_sec']:.2f}s  status={res['status']}  year={year}  singers={len(real)}  "
                            f"rag_rows={rag['rows_injected']}/{rag['rows_available']}  complete={rag['evidence_complete']}  "
                            f"text_chars={res.get('text_chars')}  thinking_chars={res.get('thinking_chars')}  "
                            f"usage={res['usage']}  cost_usd={res.get('cost_usd')}")
                if deg:
                    logger.warning(f"  Possible degenerate output ({deg_why}) - flagged only, text kept")
                if bad:
                    logger.warning(f"  TRUNCATED at the output cap (raw files: {res.get('raw_files')})")
            except Exception as e:
                logger.error(f"Query_ID={q_id}: {e}")
                bad = True
                err: Dict[str, Any] = {
                    "Query_ID": q_id, "Song_Title": song_title, "Song_ID": song_id,
                    "mode": self.mode, "strategy": self.strategy, "rag_tier": self.rag_tier,
                    "system_mode": "n/a", "system_mode_requested": self.client.system_mode_requested,
                    "rag_evidence_injected": False, "rag_resolved_year": "", "model_alias": self.model_alias,
                    "model_id": self.model_id, "model_family": self.family,
                    "Generated_Response": f"ERROR: {e}", "Parsed_Rows": [], "ReleaseYear_Parsed": "",
                    "Singer_Parsed": "ERROR", "Birthplace_Parsed": "", "PopCount_Parsed": "",
                    "Degenerate_Flag": False, "Degenerate_Reason": "", "Parse_Text_Altered": False,
                    "generation_status": "error", "generation_time_seconds": 0.0,
                    "processing_timestamp": datetime.now().isoformat(),
                    "prompt_builder": "self_ask" if self.strategy == STRATEGY_SELF_ASK else "strategy_scaffolded_v1",
                }
                err.update(self._latent_extractor.extract(None))
                results.append(err)
            self._save(results)                                         # calls are slow: checkpoint after every query
            bad_streak = bad_streak + 1 if bad else 0
            if bad_streak >= MAX_CONSECUTIVE_BAD:
                logger.error(f"{bad_streak} consecutive errors / output-cap hits - stopping this cell to protect the "
                             f"credit. Finished queries are saved; inspect raw_calls/, then rerun (resume skips the "
                             f"finished ones; truncated ones are re-run only with a larger --max_new_tokens).")
                break
        self._save(results)
        errors = sum(1 for r in results if str(r.get("Generated_Response", "")).startswith("ERROR"))
        incomplete = sum(1 for r in results if self.rag_tier != RAG_TIER_NO_EVIDENCE
                         and r.get("rag_evidence_injected") and not r.get("rag_evidence_complete"))
        logger.info("=" * 80)
        logger.info(f"SUMMARY  model={self.model_alias}  strategy={self.strategy}  rag={self.rag_tier}  "
                    f"new={new_count}  previously_done={skipped_count}  total={len(results)}  errors={errors}  "
                    f"rag_incomplete={incomplete}")
        logger.info(f"  output -> {self.output_file}")
        logger.info("=" * 80)
        return results


# ---------------------------------------------------------------------------
# REPARSE - re-run the parser over STORED response text (no Claude calls)
# ---------------------------------------------------------------------------
def reparse_outputs(root: Path = OUTPUT_ROOT) -> int:
    n = 0
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Reparse: cannot read {jf} ({e})")
            continue
        changed = False
        for r in data.get("results", []):
            txt = str(r.get("Generated_Response", ""))
            if not txt or txt.startswith("ERROR") or txt.startswith("DEGENERATE_OUTPUT"):
                continue
            parsed, altered = parse_for_strategy(txt, r.get("strategy", ""))
            deg, why = detect_degenerate(txt)
            r.update({"Parsed_Rows": parsed, "Parse_Text_Altered": altered, "Degenerate_Flag": deg, "Degenerate_Reason": why,
                      "ReleaseYear_Parsed": parsed[0].get("ReleaseYear", "") if parsed else "",
                      "Singer_Parsed": "; ".join(x["Singer"] for x in parsed),
                      "Birthplace_Parsed": "; ".join(x["Birthplace"] for x in parsed),
                      "PopCount_Parsed": "; ".join(x["PopCount"] for x in parsed)})
            n += 1
            changed = True
        if changed:
            jf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Reparse: {n} stored response(s) re-parsed under {root} (no model calls).")
    return n


# ---------------------------------------------------------------------------
# ANALYSIS & REPORTING - computed only from the saved outputs (ground truth optional)
# ---------------------------------------------------------------------------
def collect_records(root: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for jf in sorted(root.glob("*/*.json")):
        if jf.parent.name.startswith(ANALYSIS_DIR_NAME):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Analysis: cannot read {jf} ({e})")
            continue
        md = data.get("metadata", {}) or {}
        for r in data.get("results", []):
            r = dict(r)
            r.setdefault("model_alias", md.get("model_alias"))
            r.setdefault("strategy", md.get("strategy"))
            r.setdefault("rag_tier", md.get("rag_tier"))
            r.setdefault("system_mode_requested", md.get("system_mode_requested", "neutral"))
            recs.append(r)
    return recs


def _gt_singers(gt_store: Optional[GroundTruthStore], sid: str) -> Optional[List[Dict[str, Any]]]:
    if gt_store is None:
        return None
    recs = gt_store.all_for(sid)
    if not recs or any(is_error_row(r.author) for r in recs):
        return None
    merged: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if is_sentinel_singer(r.author):
            continue
        d = merged.setdefault(name_key(r.author), {"singer": r.author, "birthplace": r.birthplace, "pop": gt_pop_value(r.pop_raw)})
        if not d["birthplace"]:
            d["birthplace"] = r.birthplace
        if d["pop"] is None:
            d["pop"] = gt_pop_value(r.pop_raw)
    return list(merged.values())


def _match_pairs(g_list: List[Dict[str, Any]], p_list: List[Dict[str, Any]]):
    rem_gt, unm, pairs = list(range(len(g_list))), list(range(len(p_list))), []
    for exact in (True, False):
        for gi in list(rem_gt):
            for pi in list(unm):
                a, b = g_list[gi]["singer"], p_list[pi]["Singer"]
                if (name_key(a) == name_key(b)) if exact else names_match(a, b):
                    pairs.append((gi, pi)); rem_gt.remove(gi); unm.remove(pi)
                    break
    return pairs, rem_gt, unm


def evaluate_record(rec: Dict[str, Any], gt_store: Optional[GroundTruthStore]) -> Dict[str, Any]:
    resp = str(rec.get("Generated_Response", ""))
    is_err = resp.startswith("ERROR")
    rows = [] if is_err else (rec.get("Parsed_Rows") or [])
    pred = [r for r in rows if not is_sentinel_singer(r.get("Singer", ""))]
    sid = str(rec.get("Song_ID", ""))
    gt_year = gt_store.year_for(sid) if gt_store is not None else ""
    yp = str(rec.get("ReleaseYear_Parsed", ""))
    out: Dict[str, Any] = {
        "model_alias": rec.get("model_alias"), "strategy": rec.get("strategy"), "rag_tier": rec.get("rag_tier"),
        "system_mode": rec.get("system_mode_requested", "neutral"), "Query_ID": rec.get("Query_ID"), "Song_ID": sid,
        "status": rec.get("generation_status", ""), "is_error": is_err,
        "truncated": rec.get("generation_status") in (ST_TRUNC_LENGTH, ST_TRUNC_REASONING),
        "empty": rec.get("generation_status") == ST_EMPTY, "degenerate": bool(rec.get("Degenerate_Flag")),
        "latency": rec.get("generation_time_seconds"), "completion_tokens": _completion_tokens(rec.get("usage")),
        "text_chars": rec.get("text_chars"), "thinking_chars": rec.get("thinking_chars"),
        "cost_usd": rec.get("cost_usd_reported"), "year_parsed": yp, "gt_year": gt_year,
        "year_ok": (yp == gt_year) if (yp and gt_year) else np.nan,
        "n_pred": len(pred), "no_result": bool(rows) and not pred,
        "pop_numeric_n": sum(str(r.get("PopCount", "")).isdigit() for r in pred), "pop_total_n": len(pred),
        "tp": 0, "n_gt": 0, "scored": False, "birth_ok": 0, "birth_n": 0, "pop_5_n": 0, "pop_cmp_n": 0,
        "fn_names": [], "fp_names": [],
    }
    g_list = None if is_err else _gt_singers(gt_store, sid)
    if g_list is None:
        return out
    pairs, rem_gt, unm = _match_pairs(g_list, pred)
    out.update({"tp": len(pairs), "n_gt": len(g_list), "scored": True,
                "fn_names": [g_list[i]["singer"] for i in rem_gt], "fp_names": [pred[i]["Singer"] for i in unm]})
    for gi, pi in pairs:
        g, p = g_list[gi], pred[pi]
        out["birth_n"] += 1
        out["birth_ok"] += int(places_match(p.get("Birthplace", ""), g["birthplace"]))
        pp = str(p.get("PopCount", "")).strip()
        if pp.isdigit() and g["pop"]:
            out["pop_cmp_n"] += 1
            out["pop_5_n"] += int(abs(int(pp) - g["pop"]) / g["pop"] <= 0.05)
    return out


def _agg(g: pd.DataFrame) -> Dict[str, Any]:
    sc = g[g["scored"]]
    tp, npred, ngt = sc["tp"].sum(), sc["n_pred"].sum(), sc["n_gt"].sum()
    p = tp / npred if npred else np.nan
    r = tp / ngt if ngt else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan

    def rate(a: str, b: str) -> float:
        d = g[b].sum()
        return g[a].sum() / d if d else np.nan

    return {"records": len(g), "error_rate": g["is_error"].mean(), "truncated_rate": g["truncated"].mean(),
            "empty_rate": g["empty"].mean(), "degenerate_rate": g["degenerate"].mean(),
            "no_result_rate": g["no_result"].mean(), "mean_singers": g["n_pred"].mean(),
            "year_accuracy": pd.to_numeric(g["year_ok"], errors="coerce").mean(),
            "pop_numeric_rate": rate("pop_numeric_n", "pop_total_n"),
            "gt_scored": int(g["scored"].sum()), "gt_precision": p, "gt_recall": r, "gt_f1": f,
            "gt_birthplace_acc": rate("birth_ok", "birth_n"), "gt_pop_within_5pct": rate("pop_5_n", "pop_cmp_n"),
            "mean_latency_s": pd.to_numeric(g["latency"], errors="coerce").mean(),
            "mean_completion_tokens": g["completion_tokens"].mean(),
            "mean_text_chars": pd.to_numeric(g["text_chars"], errors="coerce").mean(),
            "mean_thinking_chars": pd.to_numeric(g["thinking_chars"], errors="coerce").mean(),
            "total_cost_usd": pd.to_numeric(g["cost_usd"], errors="coerce").sum()}


def summarise(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(by, key))
        d.update(_agg(g))
        rows.append(d)
    return pd.DataFrame(rows)


def _tbl(d: pd.DataFrame) -> str:
    return "(no data)" if d is None or d.empty else d.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def _make_plots(df: pd.DataFrame, outdir: Path, has_gt: bool) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable ({e}) - plots skipped")
        return []
    saved: List[str] = []

    def save(fig, name: str) -> None:
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    def code(s: str) -> str:
        return STRATEGY_SUFFIX.get(s, s).split("_")[0]

    bs = summarise(df, ["strategy"]).assign(code=lambda d: d["strategy"].map(code))
    bm = summarise(df, ["model_alias"])
    if not bs.empty:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(bs))
        ok = 1 - bs["truncated_rate"] - bs["empty_rate"] - bs["error_rate"]
        bottom = np.zeros(len(bs))
        for vals, lab in ((ok.clip(lower=0), "ok"), (bs["truncated_rate"], "truncated at cap"),
                          (bs["empty_rate"], "empty"), (bs["error_rate"], "error")):
            ax.bar(x, vals.values, bottom=bottom, label=lab); bottom += vals.values
        ax.set_xticks(x); ax.set_xticklabels(bs["code"]); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        ax.set_title("Call outcomes by strategy (share of records)")
        save(fig, "outcomes_by_strategy.png")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(bs["code"], bs["mean_singers"], color="#2a6fbb")
        ax.set_ylabel("mean singers parsed per song"); ax.set_title("Answer size by strategy")
        save(fig, "singers_by_strategy.png")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(bs["code"], bs["mean_latency_s"], color="#c0392b")
        ax.set_ylabel("mean latency per call (s)"); ax.set_title("Latency by strategy")
        save(fig, "latency_by_strategy.png")
    if not bm.empty:
        fig, ax = plt.subplots(figsize=(9, max(3.2, 0.4 * len(bm) + 1.5)))
        y = np.arange(len(bm))
        ax.barh(y - 0.2, bm["mean_text_chars"].fillna(0), 0.4, label="visible text (chars)", color="#2a6fbb")
        ax.barh(y + 0.2, bm["mean_thinking_chars"].fillna(0), 0.4, label="thinking (chars)", color="#c0392b")
        ax.set_yticks(y); ax.set_yticklabels(bm["model_alias"]); ax.legend(fontsize=8)
        ax.set_title("Mean visible vs thinking length per call, by model")
        save(fig, "text_vs_thinking_by_model.png")
    if has_gt and df["scored"].any():
        sc = df[df["scored"]]
        for fac, fname in (("strategy", "gt_f1_by_strategy.png"), ("model_alias", "gt_f1_by_model.png")):
            s = summarise(sc, [fac]).sort_values("gt_f1")
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.36 * len(s) + 1.5)))
            y, hh = np.arange(len(s)), 0.27
            labels = [code(v) if fac == "strategy" else v for v in s[fac]]
            ax.barh(y - hh, s["gt_precision"].fillna(0), hh, label="precision")
            ax.barh(y, s["gt_recall"].fillna(0), hh, label="recall")
            ax.barh(y + hh, s["gt_f1"].fillna(0), hh, label="F1 (micro)")
            ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 1); ax.legend(loc="lower right")
            ax.set_title(f"Ground-truth scoring by {fac}")
            save(fig, fname)
    return saved


def run_analysis(root: Path = OUTPUT_ROOT) -> None:
    recs = collect_records(root)
    if not recs:
        logger.warning(f"Analysis: no result files under {root.resolve()}.")
        return
    gt_store: Optional[GroundTruthStore] = None
    try:
        gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
    except (FileNotFoundError, ValueError):
        logger.info("Analysis: no ground-truth file found - structural analysis only (no accuracy scoring).")
    df = pd.DataFrame([evaluate_record(r, gt_store) for r in recs])
    df = df.drop_duplicates(subset=["model_alias", "strategy", "rag_tier", "system_mode", "Query_ID"], keep="last").reset_index(drop=True)
    outdir = root / ANALYSIS_DIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    has_gt = bool(df["scored"].any())
    df.drop(columns=["fn_names", "fp_names"]).to_csv(outdir / "eval_per_record.csv", index=False, encoding="utf-8-sig")
    by_model, by_strat = summarise(df, ["model_alias"]), summarise(df, ["strategy"])
    by_rag = summarise(df, ["rag_tier"])
    by_cfg = summarise(df, ["model_alias", "strategy", "rag_tier", "system_mode"])
    for name, t in (("summary_by_model", by_model), ("summary_by_strategy", by_strat),
                    ("summary_by_rag_tier", by_rag), ("summary_by_config", by_cfg)):
        t.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")
    plots = _make_plots(df, outdir, has_gt)
    h = _agg(df)
    L: List[str] = ["=" * 100, "MUSIC SINGER ANALYSER (CLAUDE CODE) - ANALYSIS REPORT",
                    f"generated {datetime.now().isoformat(timespec='seconds')}  (script version {SCRIPT_VERSION})", "=" * 100,
                    "\n1. SCOPE",
                    f"  records={len(df)}  models={df['model_alias'].nunique()}  strategies={df['strategy'].nunique()}  "
                    f"rag_tiers={df['rag_tier'].nunique()}  system modes={sorted(df['system_mode'].unique())}",
                    "  Backend: `claude -p` in a Claude Code cloud session; temperature / top_p not controllable; tools disabled; "
                    "raw events per call in <cell dir>/raw_calls.",
                    f"  Ground-truth scoring: {'ON (' + str(int(df['scored'].sum())) + ' scored records; a song with no GT rows, or with a GT retrieval error, is not scored)' if has_gt else 'OFF'}",
                    "\n2. HEADLINE (all records pooled)"]
    for k, v in h.items():
        L.append(f"  {k:<26}{v:.4f}" if isinstance(v, (float, np.floating)) and v == v else f"  {k:<26}{v}")
    L += ["\n3. BY MODEL", _tbl(by_model), "\n4. BY STRATEGY", _tbl(by_strat.assign(code=by_strat['strategy'].map(lambda s: STRATEGY_SUFFIX.get(s, s).split('_')[0]))),
          "\n5. BY RAG TIER", _tbl(by_rag), "\n6. GENERATION HEALTH",
          _tbl(df.groupby(["model_alias", "system_mode", "status"]).size().reset_index(name="count"))]
    if has_gt:
        sc = df[df["scored"]]
        miss = Counter(n for lst in sc["fn_names"] for n in lst).most_common(15)
        fps = Counter(n for lst in sc["fp_names"] for n in lst).most_common(15)
        L += ["\n7. GROUND-TRUTH DETAIL",
              "  Most missed singers: " + ("; ".join(f"{a}: {b}" for a, b in miss) or "none"),
              "  Most frequent false positives: " + ("; ".join(f"{a}: {b}" for a, b in fps) or "none")]
    ins: List[str] = []
    if h["truncated_rate"] == h["truncated_rate"] and h["truncated_rate"] > 0:
        tm = by_model[by_model["truncated_rate"] > 0].sort_values("truncated_rate", ascending=False)
        ins.append(f"{h['truncated_rate']:.1%} of calls hit the output-token cap (" +
                   ", ".join(f"{r['model_alias']} {r['truncated_rate']:.0%}" for _, r in tm.head(6).iterrows()) +
                   "). Compare text_chars with thinking_chars to see what used the budget; raise --max_new_tokens or set --max_thinking_tokens.")
    if has_gt and h["gt_f1"] == h["gt_f1"]:
        bf = by_strat.dropna(subset=["gt_f1"]).sort_values("gt_f1")
        if len(bf) > 1:
            ins.append(f"Strategy F1 spread: best {STRATEGY_SUFFIX[bf.iloc[-1]['strategy']].split('_')[0]} ({bf.iloc[-1]['gt_f1']:.3f}), "
                       f"worst {STRATEGY_SUFFIX[bf.iloc[0]['strategy']].split('_')[0]} ({bf.iloc[0]['gt_f1']:.3f}).")
    if h["year_accuracy"] == h["year_accuracy"]:
        ins.append(f"Release-year accuracy against the GT year: {h['year_accuracy']:.1%} (over records where both years are known).")
    if h["total_cost_usd"] and h["total_cost_usd"] == h["total_cost_usd"]:
        ins.append(f"Reported cost summed over all records (Claude Code 'total_cost_usd'): {h['total_cost_usd']:.2f} USD.")
    L += ["\n8. KEY INSIGHTS"] + [f"  - {s}" for s in ins]
    L += ["\n9. LIMITATIONS",
          "  - One call per (model, strategy, RAG tier, song): small differences may be sampling noise (no replicates).",
          "  - Singer matching uses normalised names with token-subset tolerance; a shared surname can merge two singers.",
          "  - Sampling parameters cannot be set through `claude -p`; a call that hits the cap is stored truncated, not retried.",
          "  - system_mode neutral vs default changes the system prompt; do not pool them when comparing models.",
          "\nFILES: " + (", ".join(Path(p).name for p in plots) if plots else "(no plots written)")]
    (outdir / "analysis_report.txt").write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Analysis written to {outdir} ({len(plots)} plots, report + CSVs)")
    print("\n".join(L[:22]))
    print(f"\n  Full report: {outdir / 'analysis_report.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    global INPUT_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE
    p = argparse.ArgumentParser(
        description=(f"Music Singer Analyser - Claude Code edition (script version {SCRIPT_VERSION})\n"
                     "Claude models only; every call is a streamed headless `claude -p` run inside a Claude Code cloud session.\n"
                     "Compositional prompting (15 strategies) x RAG (7 tiers). Raw events of every call are saved under\n"
                     "<cell dir>/raw_calls; the full log is <output root>/run.log.\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(["", "Models (--model; several aliases or 'all')", "-" * 42]
                         + [f"  {a:<20} -> {MODEL_REGISTRY[a]}{'  [DEFAULT]' if a == DEFAULT_MODEL_ALIAS else ''}" for a in sorted(MODEL_REGISTRY)]
                         + ["", "RAG tiers: " + ", ".join(RAG_TIERS), "Output root: " + str(OUTPUT_ROOT)]))
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL_ALIAS], choices=sorted(MODEL_REGISTRY) + ["all"])
    p.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default=None)
    p.add_argument("--rag_tier", default=RAG_TIER_NO_EVIDENCE, choices=RAG_TIERS)
    p.add_argument("--claude_bin", default="claude", help="Claude Code CLI executable (default: claude)")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="seconds allowed per `claude -p` call")
    p.add_argument("--max_thinking_tokens", type=int, default=None,
                   help="sets MAX_THINKING_TOKENS for the call (0 = try to disable thinking); default: untouched")
    p.add_argument("--system_mode", choices=["neutral", "default"], default="neutral",
                   help="neutral: a short neutral --system-prompt (default); default: no flag, Claude Code's own system prompt")
    p.add_argument("--no-resume", dest="no_resume", action="store_true", default=False)
    p.add_argument("--input_file", default=INPUT_FILE)
    p.add_argument("--reference_file", default=REFERENCE_FILE)
    p.add_argument("--ground_truth_file", default=GROUND_TRUTH_FILE)
    p.add_argument("--rag_max_rows", type=int, default=RAG_MAX_ROWS)
    p.add_argument("--max_rows", type=int, default=None, help="first N songs only (pilot run)")
    p.add_argument("--max_new_tokens", type=int, default=GENERATION_MAX_NEW_TOKENS,
                   help="output-token cap per call (CLAUDE_CODE_MAX_OUTPUT_TOKENS)")
    p.add_argument("--leak_policy", choices=["exclude", "keep"], default="exclude")
    p.add_argument("--reparse", action="store_true", default=False, help="re-parse STORED responses (no Claude calls), then analyse")
    p.add_argument("--analyze_only", action="store_true", default=False)
    p.add_argument("--no_analysis", action="store_true", default=False)
    args = p.parse_args()
    INPUT_FILE, REFERENCE_FILE, GROUND_TRUTH_FILE = args.input_file, args.reference_file, args.ground_truth_file

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)                 # full log of every run, next to the results
    fh = logging.FileHandler(OUTPUT_ROOT / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"=== run started: {' '.join(sys.argv)} (script version {SCRIPT_VERSION}) ===")

    if args.reparse:
        reparse_outputs(OUTPUT_ROOT)
        run_analysis(OUTPUT_ROOT)
        return 0
    if args.analyze_only:
        run_analysis(OUTPUT_ROOT)
        return 0
    if not args.strategy:
        p.error("--strategy is required unless --analyze_only or --reparse is used")
    if not shutil.which(args.claude_bin):
        print(f"ERROR: '{args.claude_bin}' (Claude Code CLI) not found on PATH. This script only runs through Claude Code.")
        return 1
    set_vars = [v for v in _BILLING_ENV_VARS if os.environ.get(v)]
    if set_vars:
        logger.warning(f"Environment variable(s) {set_vars} are set: the Claude Code CLI may bill through them instead of "
                       f"the cloud-session credit.")

    aliases = sorted(MODEL_REGISTRY) if "all" in args.model else list(dict.fromkeys(args.model))
    strategies_to_run = INDIVIDUAL_STRATEGIES if args.strategy == STRATEGY_ALL else [args.strategy]
    uses_rag = args.rag_tier != RAG_TIER_NO_EVIDENCE

    print("=" * 72)
    print(f"MUSIC SINGER ANALYSER - CLAUDE CODE   (script version {SCRIPT_VERSION})")
    print("=" * 72)
    print(f"  Models ({len(aliases)}) : {', '.join(aliases)}")
    print(f"  Input file       : {INPUT_FILE}")
    print(f"  Strategy(ies)    : {args.strategy}")
    print(f"  RAG tier         : {args.rag_tier}")
    if uses_rag:
        print(f"  Reference file   : {REFERENCE_FILE}")
        print(f"  Ground truth file: {GROUND_TRUTH_FILE}")
        print(f"  RAG row cap      : {'none - all records injected' if not args.rag_max_rows else args.rag_max_rows}")
    print(f"  System mode      : {args.system_mode}")
    print(f"  Leak policy      : {args.leak_policy}")
    print(f"  Resume           : {'OFF (--no-resume)' if args.no_resume else 'ON (default)'}")
    print(f"  Max new tokens   : {args.max_new_tokens} per call; max thinking tokens: {args.max_thinking_tokens}; timeout {args.timeout}s")
    print(f"  Output root      : {OUTPUT_ROOT}   (log: {OUTPUT_ROOT / 'run.log'}; raw events: <cell dir>/raw_calls)")
    print("=" * 72)

    try:
        df = load_input_data(INPUT_FILE, args.leak_policy, args.max_rows)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    ref_store: Optional[ReferenceStore] = None
    gt_store: Optional[GroundTruthStore] = None
    if uses_rag:
        try:
            if args.rag_tier in REF_TIERS:
                ref_store = ReferenceStore(REFERENCE_FILE)
                try:
                    gt_store = GroundTruthStore(GROUND_TRUTH_FILE)      # optional here: only used to resolve the year
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Ground truth not loaded ({e}); the input file's year column is used instead.")
            else:
                gt_store = GroundTruthStore(GROUND_TRUTH_FILE)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            print(f"       RAG evidence files must be present in cwd or script dir to use --rag_tier '{args.rag_tier}'.")
            return 1
        report_rag_coverage(df, args.rag_tier, ref_store, gt_store, args.rag_max_rows)

    overall_start = datetime.now()
    failed: List[str] = []
    for m_idx, alias in enumerate(aliases, 1):
        print(f"\n{'#' * 72}\n  MODEL [{m_idx}/{len(aliases)}] {alias}\n{'#' * 72}")
        try:
            client = ClaudeCodeClient(model_alias=alias, max_new_tokens=args.max_new_tokens, system_mode=args.system_mode,
                                      claude_bin=args.claude_bin, timeout=args.timeout,
                                      max_thinking_tokens=args.max_thinking_tokens)
            client.raw_dir = OUTPUT_ROOT / "raw_calls"
            client.validate_access()
        except Exception as e:
            logger.error(f"{alias}: startup check failed - skipping this model ({e})")
            failed.append(f"{alias}[startup]")
            continue
        for strategy in strategies_to_run:
            print(f"\n{'-' * 72}\n  {alias} | strategy: {strategy} ({STRATEGY_SUFFIX[strategy]})"
                  f"{' | RAG: ' + args.rag_tier if uses_rag else ''}\n{'-' * 72}")
            try:
                analyser = MusicSingerAnalyser(strategy=strategy, client=client, df=df, no_resume=args.no_resume,
                                               rag_tier=args.rag_tier, ref_store=ref_store, gt_store=gt_store,
                                               rag_max_rows=args.rag_max_rows)
                t0 = datetime.now()
                results = analyser.process()
                ok = sum(1 for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR"))
                print(f"  done: total={len(results)} success={ok} errors={len(results) - ok} "
                      f"time={datetime.now() - t0}  -> {analyser.output_file}")
                if len(results) - ok or len(results) < len(df):
                    failed.append(f"{alias}[{strategy}]")
            except KeyboardInterrupt:
                print("\n  Interrupted - progress saved.")
                return 130
            except Exception as e:
                print(f"  Fatal error in {alias}/{strategy}: {e}")
                traceback.print_exc()
                failed.append(f"{alias}[{strategy}]")

    print(f"\n{'=' * 72}\nALL RUNS COMPLETE\n  Total wall-clock time : {datetime.now() - overall_start}\n"
          f"  Output root           : {OUTPUT_ROOT}")
    if failed:
        print(f"  Failed / incomplete: {failed}")
    print("=" * 72)
    if not args.no_analysis:
        try:
            run_analysis(OUTPUT_ROOT)
        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())