# -- VENDORED FROZEN INSTRUMENT (TRIMMED) -- do not edit behavior --------------
# source: experiments/fdgc/datasets.py (1,266 lines upstream)
# trim: kept ONLY the instrument-path surface, copied VERBATIM:
#   _pack_texts, _eval_doc_skip (FDGC_EVAL_DOC_SKIP), _try_load_c4_streaming,
#   load_c4_dataset, load_mixed_generic_dataset, the legal block
#   (_SYNTH_LEGAL, _try_load_pile_of_law, _try_load_lex_glue, load_legal_dataset),
#   the code block (_SYNTH_CODE, _try_load_stack_smol, _try_load_codeparrot,
#   load_code_dataset), the finance_news block, the math block, and
#   load_generic with GENERIC_LOADERS trimmed to {wikitext, c4, mixed}.
# dropped: codeheavy/domainmix/langmix/asdomain generics, MMLU/regmix/
#   c4-partition/mC4 loaders, load_domain (reaches mri_compressor.mri.*).
#   FDGC_GENERIC_DOMAIN and FDGC_C4PART_EXTRA_SKIP lived only on dropped paths.
# import rewrite: mri_compressor.data_utils -> mrilab.datasets.data_utils
# vendored: 2026-09-01 (manual trim; recorded in scripts/VENDOR_MANIFEST.json)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""
FDGC — generic and domain calibration datasets (instrument-path trim).

All loaders return a TextDataset of shape (n_chunks, max_seq_len). Each loader
has a primary source + fallbacks + a synthetic last resort so experiments
don't crash when HF is unreachable.

Generic sources
---------------
    wikitext   : wikitext-103 (validation)
    c4         : allenai/c4 (en, validation, streamed)
    mixed      : half wikitext-103 + half C4

Domains
-------
    legal      : pile-of-law -> lex_glue (case_hold) -> synthetic
    code       : bigcode/the-stack-smol (python) -> codeparrot -> synthetic
    math       : proof-pile -> gsm8k+math -> synthetic
    finance    : financial-news-articles (running prose; no fallback)
"""

from __future__ import annotations

import os

from typing import List, Optional

import torch

from mrilab.datasets.data_utils import TextDataset, load_wikitext_data


# ---------------------------------------------------------------------------
# Shared helper: concatenate + tokenize + chunk
# ---------------------------------------------------------------------------

def _pack_texts(
    tokenizer,
    texts: List[str],
    max_seq_len: int,
    n_samples: int,
    label: str,
) -> TextDataset:
    """Concatenate, tokenize, and chunk a list of strings into a TextDataset."""
    all_text = "\n\n".join(t for t in texts if t and t.strip())
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
    n_chunks = min(n_samples, len(tokens) // max_seq_len)
    if n_chunks < 4:
        # Calibration sets sometimes have too few tokens; repeat.
        repeat = (4 * max_seq_len // max(len(tokens), 1)) + 1
        tokens = tokens.repeat(repeat)
        n_chunks = min(n_samples, len(tokens) // max_seq_len)
    chunks = tokens[: n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
    print(f"    {label}: {n_chunks} chunks of {max_seq_len} tokens")
    return TextDataset(chunks)


# ---------------------------------------------------------------------------
# Generic: C4
# ---------------------------------------------------------------------------

def _eval_doc_skip() -> int:
    """Number of qualifying documents to skip before sampling.

    Set FDGC_EVAL_DOC_SKIP=S to evaluate on documents disjoint from the
    calibration prefix (which starts at document 0), giving a genuine
    out-of-sample split for the real-domain and generic corpora. Honored by
    C4, PubMedQA, lex_glue, codeparrot, and the GSM8K math loader. Small
    corpora (e.g. the MMLU finance domain) do not have enough held-out
    documents and are excluded from the out-of-sample check.
    """
    return int(os.environ.get("FDGC_EVAL_DOC_SKIP", "0"))


def _try_load_c4_streaming(n: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/c4", "en",
            split="validation", streaming=True,
            trust_remote_code=False,
        )
        texts = []
        skip = _eval_doc_skip()
        qual = 0
        for row in ds:
            t = row.get("text", "").strip()
            if len(t) <= 100:
                continue
            qual += 1
            if qual <= skip:
                continue
            texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    C4: streamed {len(texts)} documents"
                  + (f" (eval skip={skip})" if skip else ""))
            return texts
        return None
    except Exception as e:
        print(f"    C4 load failed: {e}")
        return None


def load_c4_dataset(tokenizer, max_seq_len: int = 512, n_samples: int = 256) -> TextDataset:
    print("  Loading C4 generic dataset...")
    texts = _try_load_c4_streaming(n_samples * 4)
    if texts is None:
        print("    C4 unavailable, falling back to wikitext")
        return load_wikitext_data(
            tokenizer, split="validation",
            max_seq_len=max_seq_len, num_samples=n_samples,
        )
    return _pack_texts(tokenizer, texts, max_seq_len, n_samples, label="C4")


# ---------------------------------------------------------------------------
# Generic: mixed (half wikitext + half C4)
# ---------------------------------------------------------------------------

def load_mixed_generic_dataset(
    tokenizer, max_seq_len: int = 512, n_samples: int = 256
) -> TextDataset:
    """Mix wikitext + C4 in roughly equal token shares.

    The point is to make the 'generic' baseline distribution-robust —
    neither encyclopedia-only nor web-only — so that observed
    domain/generic separation is not a calibration artifact.
    """
    print("  Loading mixed generic (wikitext + C4)...")
    half = max(n_samples // 2, 1)
    wiki = load_wikitext_data(
        tokenizer, split="validation",
        max_seq_len=max_seq_len, num_samples=half,
    )
    c4_texts = _try_load_c4_streaming(half * 4) or []
    if not c4_texts:
        print("    C4 missing; returning wikitext-only (mix degraded).")
        return wiki

    # Tokenize C4 half with the same recipe as wiki.
    all_c4 = "\n\n".join(c4_texts)
    c4_tokens = tokenizer.encode(all_c4, return_tensors="pt")[0]
    n_c4 = min(half, len(c4_tokens) // max_seq_len)
    if n_c4 < 1:
        return wiki
    c4_chunks = c4_tokens[: n_c4 * max_seq_len].reshape(n_c4, max_seq_len)

    # Concatenate along batch dim.
    chunks = torch.cat([wiki.input_ids, c4_chunks], dim=0)
    # Shuffle so batching interleaves sources.
    perm = torch.randperm(chunks.shape[0])
    chunks = chunks[perm]
    print(f"    mixed: {chunks.shape[0]} chunks "
          f"({wiki.input_ids.shape[0]} wiki + {n_c4} c4)")
    return TextDataset(chunks)


# ---------------------------------------------------------------------------
# Domain: legal
# ---------------------------------------------------------------------------

_SYNTH_LEGAL = [
    "The plaintiff, having established a prima facie case of negligence, bears the burden of persuading the trier of fact that the defendant breached a duty of reasonable care and that such breach was the proximate cause of the alleged injury.",
    "Pursuant to Federal Rule of Civil Procedure 56, summary judgment is appropriate where there is no genuine dispute as to any material fact and the movant is entitled to judgment as a matter of law.",
    "The court held that the warrantless search of the defendant's vehicle was unconstitutional under the Fourth Amendment because the officer lacked probable cause and no exigent circumstances existed to justify the intrusion.",
    "A contract is voidable for unconscionability when its terms are so oppressive or unfair, and the bargaining process so procedurally defective, that no reasonable party would have assented to them absent duress or misinformation.",
    "Under the doctrine of stare decisis, lower courts are bound by the holdings of higher courts within the same jurisdiction, although dicta carry only persuasive weight.",
    "The statute of limitations for a personal injury claim accrues when the plaintiff knew or, with the exercise of reasonable diligence, should have known of the injury and its causal connection to the defendant's conduct.",
    "In determining whether a particular use of copyrighted material constitutes fair use under 17 U.S.C. § 107, courts weigh the purpose and character of the use, the nature of the copyrighted work, the amount used, and the effect on the potential market.",
    "A piercing of the corporate veil requires a showing of unity of interest and ownership between the corporation and its shareholder such that the separate personalities no longer exist, combined with an inequitable result if the acts are treated as those of the corporation alone.",
]


def _try_load_pile_of_law(n: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "pile-of-law/pile-of-law", "courtListener_opinions",
            split="validation", streaming=True,
            trust_remote_code=True,
        )
        texts = []
        for row in ds:
            t = row.get("text", "").strip()
            if len(t) > 500:
                texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    pile-of-law: streamed {len(texts)} opinions")
            return texts
        return None
    except Exception as e:
        print(f"    pile-of-law failed: {e}")
        return None


def _try_load_lex_glue(n: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("lex_glue", "case_hold", split="train")
        texts = []
        skip = _eval_doc_skip()
        qual = 0
        for row in ds:
            context = row.get("context", "").strip()
            if len(context) <= 200:
                continue
            qual += 1
            if qual <= skip:
                continue
            texts.append(context)
            if len(texts) >= n:
                break
        if texts:
            print(f"    lex_glue/case_hold: loaded {len(texts)} contexts"
                  + (f" (eval skip={skip})" if skip else ""))
            return texts
        return None
    except Exception as e:
        print(f"    lex_glue failed: {e}")
        return None


def load_legal_dataset(tokenizer, max_seq_len: int = 512, n_samples: int = 256) -> TextDataset:
    print("  Loading legal domain dataset...")
    texts = _try_load_pile_of_law(n_samples * 4)
    if texts is None:
        texts = _try_load_lex_glue(n_samples * 4)
    if texts is None:
        print("    Legal: using synthetic fallback")
        texts = _SYNTH_LEGAL * (max(1, n_samples * 4 // len(_SYNTH_LEGAL) + 1))
    return _pack_texts(tokenizer, texts, max_seq_len, n_samples, label="legal")


# ---------------------------------------------------------------------------
# Domain: code
# ---------------------------------------------------------------------------

_SYNTH_CODE = [
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def prepend(self, value):\n        node = Node(value)\n        node.next = self.head\n        self.head = node\n\n    def to_list(self):\n        out, cur = [], self.head\n        while cur:\n            out.append(cur.value)\n            cur = cur.next\n        return out\n",
    "import asyncio\n\nasync def fetch(url: str, session) -> dict:\n    async with session.get(url, timeout=10) as resp:\n        resp.raise_for_status()\n        return await resp.json()\n\nasync def main(urls):\n    import aiohttp\n    async with aiohttp.ClientSession() as s:\n        tasks = [fetch(u, s) for u in urls]\n        return await asyncio.gather(*tasks)\n",
    "#include <vector>\n#include <algorithm>\n\nint longest_increasing_subsequence(const std::vector<int>& nums) {\n    std::vector<int> tails;\n    for (int x : nums) {\n        auto it = std::lower_bound(tails.begin(), tails.end(), x);\n        if (it == tails.end()) tails.push_back(x);\n        else *it = x;\n    }\n    return tails.size();\n}\n",
    "function memoize(fn) {\n  const cache = new Map();\n  return (...args) => {\n    const key = JSON.stringify(args);\n    if (cache.has(key)) return cache.get(key);\n    const v = fn(...args);\n    cache.set(key, v);\n    return v;\n  };\n}\n",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\nWHERE order_date >= DATE '2024-01-01'\nGROUP BY customer_id\nHAVING SUM(amount) > 1000\nORDER BY total DESC\nLIMIT 50;\n",
]


def _try_load_stack_smol(n: int, skip: int = 0) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "bigcode/the-stack-smol", data_dir="data/python",
            split="train", trust_remote_code=False,
        )
        texts = []
        qualifying = 0
        for row in ds:
            t = row.get("content", "").strip()
            if len(t) <= 200:
                continue
            qualifying += 1
            if qualifying <= skip:
                continue
            texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    the-stack-smol/python: loaded {len(texts)} files"
                  + (f" (skip={skip})" if skip else ""))
            return texts
        return None
    except Exception as e:
        print(f"    the-stack-smol failed: {e}")
        return None


def _try_load_codeparrot(n: int, skip: Optional[int] = None) -> Optional[List[str]]:
    if skip is None:
        skip = _eval_doc_skip()
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "codeparrot/codeparrot-clean-valid",
            split="train", streaming=True,
        )
        texts = []
        qualifying = 0
        for row in ds:
            t = row.get("content", "").strip()
            if len(t) <= 200:
                continue
            qualifying += 1
            if qualifying <= skip:
                continue
            texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    codeparrot-clean-valid: streamed {len(texts)} files"
                  + (f" (skip={skip})" if skip else ""))
            return texts
        return None
    except Exception as e:
        print(f"    codeparrot failed: {e}")
        return None


def load_code_dataset(tokenizer, max_seq_len: int = 512, n_samples: int = 256) -> TextDataset:
    print("  Loading code domain dataset...")
    texts = _try_load_stack_smol(n_samples * 4)
    if texts is None:
        texts = _try_load_codeparrot(n_samples * 4)
    if texts is None:
        print("    Code: using synthetic fallback")
        texts = _SYNTH_CODE * (max(1, n_samples * 4 // len(_SYNTH_CODE) + 1))
    return _pack_texts(tokenizer, texts, max_seq_len, n_samples, label="code")


# ---------------------------------------------------------------------------
# Domain: finance_news (running financial prose)
#
# Format control for the finance domain. The default `finance` corpus is
# MMLU multiple-choice questions; `finance_news` is running financial-news
# prose (Reuters/financial articles), holding the finance *topic* fixed while
# changing the *format*. Used to show the domain gate is not an MCQ-format
# artifact: rebuilding V_joint with finance_news in place of finance
# reproduces the gate. Parquet-based (loads under datasets>=4), non-gated.
# ---------------------------------------------------------------------------

def _try_load_financial_news(n: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("ashraq/financial-news-articles",
                          split="train", streaming=True)
        texts = []
        skip = _eval_doc_skip()
        qual = 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) <= 200:
                continue
            qual += 1
            if qual <= skip:
                continue
            texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    financial-news-articles: streamed {len(texts)} articles"
                  + (f" (eval skip={skip})" if skip else ""))
            return texts
        return None
    except Exception as e:
        print(f"    financial-news-articles load failed: {e}")
        return None


def load_finance_news_dataset(tokenizer, max_seq_len: int = 512,
                              n_samples: int = 256) -> TextDataset:
    print("  Loading finance_news (running financial prose)...")
    texts = _try_load_financial_news(n_samples * 4)
    if texts is None:
        raise RuntimeError("finance_news: financial-news-articles unavailable "
                           "(no fallback: this is a format control and must be "
                           "running prose)")
    return _pack_texts(tokenizer, texts, max_seq_len, n_samples,
                       label="finance_news")


# ---------------------------------------------------------------------------
# Domain: math
# ---------------------------------------------------------------------------

_SYNTH_MATH = [
    "Theorem. Let G be a finite group of order n and H a subgroup of G. Then the order of H divides n. Proof. By Lagrange's theorem, the left cosets of H partition G into |G|/|H| equivalence classes of equal size |H|, hence |H| | |G|.",
    "Let f: R -> R be twice differentiable with f''(x) > 0 for all x. Then f is strictly convex, and for any a < b the secant line from (a, f(a)) to (b, f(b)) lies strictly above the graph of f on (a, b).",
    "The Riemann zeta function zeta(s) = sum_{n=1}^infty 1/n^s converges absolutely for Re(s) > 1 and admits an analytic continuation to C \\ {1} with a simple pole of residue 1 at s=1.",
    "Problem. Evaluate the integral of x * exp(-x^2) from 0 to infinity. Solution. Substitute u = x^2, du = 2x dx, so the integral becomes (1/2) integral of exp(-u) du from 0 to infinity = 1/2.",
    "Let V be a finite-dimensional vector space over a field K and T: V -> V a linear operator. If T is diagonalizable then the minimal polynomial of T splits into distinct linear factors over K.",
    "A metric space (X, d) is compact if and only if every sequence in X has a convergent subsequence. In R^n with the Euclidean metric this is equivalent to being closed and bounded.",
    "The Gaussian integral integral_{-infty}^{infty} exp(-x^2) dx = sqrt(pi). Squaring and switching to polar coordinates gives integral integral exp(-r^2) r dr dtheta over [0, 2pi) x [0, infty) = pi.",
    "Prove that sqrt(2) is irrational. Suppose sqrt(2) = p/q in lowest terms. Then 2 q^2 = p^2, so p is even, p = 2k. Then q^2 = 2 k^2, so q is even. Contradiction with gcd(p,q) = 1.",
]


def _try_load_proof_pile(n: int) -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "hoskinson-center/proof-pile",
            split="train", streaming=True,
        )
        texts = []
        for row in ds:
            t = row.get("text", "").strip()
            if len(t) > 300:
                texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            print(f"    proof-pile: streamed {len(texts)} docs")
            return texts
        return None
    except Exception as e:
        print(f"    proof-pile failed: {e}")
        return None


def _try_load_math_mix(n: int) -> Optional[List[str]]:
    """GSM8K solutions + MATH problems concatenated as math calibration."""
    try:
        from datasets import load_dataset
        texts = []
        skip = _eval_doc_skip()
        try:
            gsm = load_dataset("gsm8k", "main", split="train")
            qual = 0
            for row in gsm:
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                if not (q and a):
                    continue
                qual += 1
                if qual <= skip:
                    continue
                texts.append(f"{q}\n{a}")
                if len(texts) >= n // 2:
                    break
        except Exception as e:
            print(f"    gsm8k failed: {e}")
        try:
            mth = load_dataset("hendrycks/competition_math", split="train")
            for row in mth:
                p = row.get("problem", "").strip()
                s = row.get("solution", "").strip()
                if p and s:
                    texts.append(f"{p}\n{s}")
                if len(texts) >= n:
                    break
        except Exception as e:
            print(f"    competition_math failed: {e}")
        if texts:
            print(f"    math mix: loaded {len(texts)} problems")
            return texts
        return None
    except Exception as e:
        print(f"    math mix failed: {e}")
        return None


def load_math_dataset(tokenizer, max_seq_len: int = 512, n_samples: int = 256) -> TextDataset:
    print("  Loading math domain dataset...")
    texts = _try_load_proof_pile(n_samples * 4)
    if texts is None:
        texts = _try_load_math_mix(n_samples * 4)
    if texts is None:
        print("    Math: using synthetic fallback")
        texts = _SYNTH_MATH * (max(1, n_samples * 4 // len(_SYNTH_MATH) + 1))
    return _pack_texts(tokenizer, texts, max_seq_len, n_samples, label="math")


# ---------------------------------------------------------------------------
# Dispatch tables (trimmed: only loaders kept above)
# ---------------------------------------------------------------------------

GENERIC_LOADERS = {
    "wikitext": lambda tok, msl, ns: load_wikitext_data(
        tok, split="validation", max_seq_len=msl, num_samples=ns,
    ),
    "c4": load_c4_dataset,
    "mixed": load_mixed_generic_dataset,
}


def load_generic(name: str, tokenizer, max_seq_len: int, n_samples: int) -> TextDataset:
    if name not in GENERIC_LOADERS:
        raise ValueError(
            f"Unknown generic '{name}'. Supported: {sorted(GENERIC_LOADERS)}"
        )
    return GENERIC_LOADERS[name](tokenizer, max_seq_len, n_samples)
