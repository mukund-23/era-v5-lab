"""Shared setup for teams/04-ke2/tests.

* Puts teams/04-ke2/src/tables and teams/04-ke2/src/collisions on sys.path so tests can
  ``import build_tables`` / ``import build_collisions`` / ``import inspect_tokenizer``.
* Builds the tokenizer bundle and the four byte tables ONCE per pytest run
  (session scope) — building them takes ~1 min on CPU.

Tokenizer: ``$KE2_TOKENIZER`` (path to tokenizer.json or its folder, or an HF id),
else build_tables' default (teams/04-ke2/data/tokenizer.json, then the HF id
theschoolofai/BrahmicTokenizer-131K).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Aliases: pr-guard's email check misreads dotted pytest decorators as addresses.
fixture = pytest.fixture

TEAM_DIR = Path(__file__).resolve().parent.parent          # teams/04-ke2
for sub in ("src/tables", "src/collisions"):
    p = str(TEAM_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import build_tables as bt  # noqa: E402

# sha256 of BrahmicTokenizer-131K tokenizer.json (main, as of 2026-10-25).
# Tests that pin exact counts only run against this file.
BRAHMIC_SHA256 = "01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff"


@fixture(scope="session")
def tb():
    return bt.load_tokenizer(os.environ.get("KE2_TOKENIZER"))


@fixture(scope="session")
def tables(tb):
    """{(kind, P): ByteTable} for kind in (decode, raw) and P in bt.POS_DIMS."""
    out = {}
    for P in bt.POS_DIMS:
        out["decode", P] = bt.build_decode_table(tb, P)
        out["raw", P] = bt.build_raw_table(tb, P)
    return out


@fixture(scope="session")
def raw32(tables):
    return tables["raw", 32]


@fixture(scope="session")
def dec32(tables):
    return tables["decode", 32]


@fixture(scope="session")
def is_brahmic(tb):
    """True only for the exact BrahmicTokenizer-131K file the pinned counts came from."""
    return tb.sha256 == BRAHMIC_SHA256
