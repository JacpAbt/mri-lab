# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/prefs.py
# source sha256: c66a29a26ddb018a13d212d7e4430cfd6e3616c772eb4ac07edf2a6f6227d898
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""Paired preference items, and the two constructed anchors the shape statistics
are read against.

Every item is `(prompt, preferred, dispreferred)` where both continuations follow
the *same* prompt. That is the whole point: p(x) is held exactly fixed, so
anything the instrument finds sits in the conditional rather than in the corpus.
It also means the minimal-pair objection that sinks the corpus-level disposition
result (`domain_locus/where_disp_*`, shared 0.99 on inputs that barely differ)
cannot apply here — identical prompts are the design, not a confound.

Four sources are real dispositions; three are anchors with known shape, and the
anchors are not optional. "Effective rank 40" is uninterpretable without knowing
what this estimator reports for an object whose shape you already know.

  spelling   a pure unigram-frequency preference. Cannot be a gate. Sets the
             tilt end of the bracket.
  grammar    a deterministic rule with a large margin. Sets the "high margin but
             not necessarily low rank" corner, which is the one that would
             otherwise let a high-margin tilt be misread as a gate.
  (gate)     constructed at runtime by install_gate.py, not here.

No-network by construction for the anchors: a rig check that needs the Hub is a
rig check that fails on a bad day.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

Item = Tuple[str, str, str]

# (British, American) frequency variants, grouped by word class. The grouping is
# load-bearing: dropping "realise" into a noun slot ("noticed the realise") turns
# the anchor into a grammaticality test, and a grammaticality preference is not a
# frequency tilt. Ambiguous pairs are excluded rather than disambiguated --
# licence/license and practise/practice split noun from verb differently in the
# two dialects, so either slot makes them a syntax item.
_SPELLING_NOUNS = [
    ("colour", "color"), ("favour", "favor"), ("honour", "honor"),
    ("labour", "labor"), ("neighbour", "neighbor"), ("behaviour", "behavior"),
    ("flavour", "flavor"), ("humour", "humor"), ("rumour", "rumor"),
    ("armour", "armor"), ("harbour", "harbor"), ("vapour", "vapor"),
    ("centre", "center"), ("theatre", "theater"), ("litre", "liter"),
    ("metre", "meter"), ("fibre", "fiber"), ("defence", "defense"),
    ("offence", "offense"), ("catalogue", "catalog"), ("dialogue", "dialog"),
    ("programme", "program"), ("aluminium", "aluminum"), ("plough", "plow"),
]
_SPELLING_VERBS = [
    ("organise", "organize"), ("realise", "realize"), ("analyse", "analyze"),
    ("apologise", "apologize"), ("recognise", "recognize"),
    ("criticise", "criticize"), ("emphasise", "emphasize"),
    ("summarise", "summarize"), ("prioritise", "prioritize"),
    ("standardise", "standardize"),
]
_SPELLING_PARTICIPLES = [
    ("travelled", "traveled"), ("cancelled", "canceled"),
    ("modelled", "modeled"), ("labelled", "labeled"),
    ("signalled", "signaled"), ("fuelled", "fueled"),
]
_SPELLING_ADJECTIVES = [
    ("marvellous", "marvelous"), ("grey", "gray"), ("sombre", "somber"),
    ("sceptical", "skeptical"),
]

_SPELLING_SLOTS = (
    (_SPELLING_NOUNS, [
        "The report described the",
        "She wrote at length about the",
        "Everyone in the room noticed the",
        "The article opened with a note on the",
        "He could not stop thinking about the",
        "Their main complaint concerned the",
        "The committee finally agreed on the",
        "Nothing in the file explained the",
    ]),
    (_SPELLING_VERBS, [
        "They decided to",
        "The team was asked to",
        "It would be sensible to",
        "Nobody had thought to",
        "The plan is to",
        "We will need to",
    ]),
    (_SPELLING_PARTICIPLES, [
        "By the end of the year they had",
        "The meeting was",
        "It turned out the flight had been",
        "For two decades the process was",
    ]),
    (_SPELLING_ADJECTIVES, [
        "The whole thing looked",
        "Her first reaction was",
        "The walls had gone",
        "Most of the committee remained",
    ]),
)

# Vowel-initial nouns: "an X" is obligatory, "a X" is a grammatical error. A rule
# the model applies with near-certainty, so a large margin with no gate anywhere.
_VOWEL_NOUNS = [
    "apple", "orange", "umbrella", "elephant", "envelope", "island", "engine",
    "argument", "idea", "answer", "opinion", "accident", "election", "invoice",
    "eagle", "oven", "alarm", "essay", "insult", "orchard", "anchor", "opening",
    "interview", "instrument", "example", "exit", "error", "offer", "onion",
    "ambulance", "airport", "author", "album", "editor", "hour", "honest man",
]

_GRAMMAR_CARRIERS = [
    "On the table she found",
    "He said it looked like",
    "They had never seen",
    "The letter arrived with",
    "Somebody left behind",
    "It turned out to be",
    "We were offered",
    "The photograph showed",
]


def _spelling(n: int, seed: int = 0) -> List[Item]:
    """Preferred = British. The sign is arbitrary; only the shape is read.

    Slots are interleaved rather than concatenated so a truncated request still
    spans all four word classes: an anchor made entirely of `-our` nouns would be
    measuring one suffix.
    """
    rng = random.Random(seed)
    buckets = []
    for words, carriers in _SPELLING_SLOTS:
        b = [(c, br, us) for c in carriers for br, us in words]
        rng.shuffle(b)
        buckets.append(b)
    out: List[Item] = []
    while len(out) < n and any(buckets):
        for b in buckets:
            if not b:
                continue
            carrier, br, us = b.pop()
            out.append((carrier, f" {br}", f" {us}"))
            if len(out) >= n:
                break
    return out


def _grammar(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    pairs = [(c, w) for c in _GRAMMAR_CARRIERS for w in _VOWEL_NOUNS]
    rng.shuffle(pairs)
    return [(carrier, f" an {w}", f" a {w}") for carrier, w in pairs[:n]]


def _security(n: int) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO",
                      split="train", streaming=True)
    items: List[Item] = []
    for row in ds:
        if row.get("chosen") and row.get("rejected") and row.get("question"):
            items.append((row["question"], row["chosen"], row["rejected"]))
        if len(items) >= n:
            break
    return items


def _truthfulness(n: int) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    items: List[Item] = []
    for row in ds:
        bad = row["incorrect_answers"][0] if row["incorrect_answers"] else None
        if row["best_answer"] and bad:
            items.append((f"Q: {row['question']}\nA:",
                          " " + row["best_answer"], " " + bad))
        if len(items) >= n:
            break
    return items


def _harmfulness(n: int) -> List[Item]:
    """Safer vs less-safe response to the same prompt.

    PKU-SafeRLHF is the primary because it is already in (prompt, response_0,
    response_1, safer_response_id) form, so no dialogue parsing is needed and the
    pairing is explicit. hh-rlhf harmless-base is the fallback, split on the last
    "Assistant:" turn.

    This is the disposition post-training most obviously targets, so it is the
    one where H-GT predicts the ladder does something. If refusal does not
    localise here, the two-regime picture has no positive case at all.
    """
    from datasets import load_dataset
    items: List[Item] = []
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train",
                          streaming=True)
        for row in ds:
            sid = row.get("safer_response_id")
            r0, r1 = row.get("response_0"), row.get("response_1")
            if sid is None or not r0 or not r1 or r0 == r1:
                continue
            good, bad = (r0, r1) if int(sid) == 0 else (r1, r0)
            items.append((str(row["prompt"]).strip(), " " + good.strip(),
                          " " + bad.strip()))
            if len(items) >= n:
                break
    except Exception as exc:                                    # noqa: BLE001
        print(f"    PKU-SafeRLHF unavailable ({exc}); falling back to hh-rlhf")
    if len(items) >= n:
        return items
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base",
                      split="train", streaming=True)
    tag = "\n\nAssistant:"
    for row in ds:
        ch, rj = row.get("chosen", ""), row.get("rejected", "")
        if tag not in ch or tag not in rj:
            continue
        stem = ch[:ch.rfind(tag) + len(tag)]
        if not rj.startswith(stem):
            continue                      # the two only differ in the last turn
        good, bad = ch[len(stem):], rj[len(stem):]
        if not good.strip() or not bad.strip() or good == bad:
            continue
        items.append((stem, good, bad))
        if len(items) >= n:
            break
    return items


# Multi-substitution dialect variants. This is the tilt anchor that survives the
# fixed-token-pair confound: `grammar` differs in the SAME token pair (" an" vs
# " a") on every item, so its late-layer margin gradient is one fixed unembedding
# difference for all items and coherence reads ~1 for a purely lexical reason.
# `spelling` is better (varied pairs) but still one token per item. Here each
# continuation carries two or three substitutions drawn from a rotating lexicon,
# so no token pair repeats across items and the preference can only be
# distributional.
_DIALECT = [
    ("lift", "elevator"), ("lorry", "truck"), ("flat", "apartment"),
    ("biscuit", "cookie"), ("petrol", "gasoline"), ("queue", "line"),
    ("rubbish", "garbage"), ("holiday", "vacation"), ("autumn", "fall"),
    ("pavement", "sidewalk"), ("jumper", "sweater"), ("chemist", "pharmacy"),
    ("torch", "flashlight"), ("boot", "trunk"), ("nappy", "diaper"),
    ("cot", "crib"), ("plaster", "bandage"), ("aubergine", "eggplant"),
    ("courgette", "zucchini"), ("motorway", "freeway"), ("postcode", "zip code"),
    ("trainers", "sneakers"), ("wardrobe", "closet"), ("tap", "faucet"),
]

# Slots that take a placeable object, and slots that take anything. Keeping them
# apart matters only for how natural the carrier reads -- both sides of a pair get
# the identical frame, so the contrast is clean either way -- but an unnatural
# carrier inflates the within-item variance the margin has to be measured against.
_DIALECT_ABSTRACT = {"autumn", "holiday", "postcode", "queue", "chemist",
                     "motorway", "pavement"}

_DIALECT_OBJECT_FRAMES = [
    "he waited by the {0}, put the {1} down, and said nothing",
    "she left the {0} near the {1} and went back inside",
    "there was a {0} beside the {1} and no sign of anyone",
    "the {0} was gone, the {1} was ruined, and it was late",
    "nobody had moved the {0} or the {1} since the morning",
]
_DIALECT_ANY_FRAMES = [
    "they argued about the {0}, the {1}, and the weather",
    "the letter mentioned the {0} twice and the {1} once",
    "nothing in the file explained the {0} or the {1}",
]


def _dialect(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    prompts = [
        "The last paragraph of the letter read:",
        "His diary entry for that day says:",
        "The witness statement continues:",
        "She wrote back the same evening:",
        "The report's closing note reads:",
    ]
    objects = [p for p in _DIALECT if p[0] not in _DIALECT_ABSTRACT]
    out: List[Item] = []
    seen = set()
    for _ in range(n * 12):
        if len(out) >= n:
            break
        if rng.random() < 0.6:
            frame, pool = rng.choice(_DIALECT_OBJECT_FRAMES), objects
        else:
            frame, pool = rng.choice(_DIALECT_ANY_FRAMES), _DIALECT
        k = frame.count("{")
        pairs = rng.sample(pool, k)
        br = frame.format(*[p[0] for p in pairs])
        us = frame.format(*[p[1] for p in pairs])
        prompt = rng.choice(prompts)
        if (prompt, br) in seen:
            continue
        seen.add((prompt, br))
        out.append((prompt, " " + br, " " + us))
    return out


# ---------------------------------------------------------------------------
# Battery sources: >=2 independent datasets per category, so a per-category
# claim can be checked for dataset-specificity the same way the domain track
# used fallback chains. Every repo below was load-tested against the
# script-refusing `datasets` regime on 2026-08-18. Sign convention throughout:
# preferred = the pro-social side (secure/safe/true/anti-stereotype/moral/
# civil/non-sycophantic); for direction geometry the sign is arbitrary anyway.
# ---------------------------------------------------------------------------

def _hf_pairs(n, repo, fn, config=None, split="train", **load_kw):
    from datasets import load_dataset

    def _load(streaming):
        args = (repo, config) if config else (repo,)
        return load_dataset(*args, split=split, streaming=streaming, **load_kw)
    # Streaming refuses script-bearing repos where the non-streaming
    # parquet-export fallback succeeds; script repos fail at load time, so
    # falling back on the constructor keeps iteration lazy either way.
    try:
        ds = _load(True)
    except Exception:                                            # noqa: BLE001
        ds = _load(False)
    items: List[Item] = []
    for row in ds:
        it = fn(row)
        if it is None:
            continue
        p, g, b = it
        if g.strip() and b.strip() and g != b:
            items.append((p, g, b))
        if len(items) >= n:
            break
    return items


def _security_sven(n):
    """Real GitHub security-fix pairs (SVEN, He & Vechev CCS'23 mirror)."""
    def fn(r):
        v, f = r.get("func_src_before", ""), r.get("func_src_after", "")
        if not (40 < len(v) < 1500 and 40 < len(f) < 1500):
            return None
        return ("", " " + f.strip(), " " + v.strip())
    return _hf_pairs(n, "bstee615/sven", fn)


def _security_cve(n):
    """CVE fix pairs; long multi-file chunks filtered out."""
    def fn(r):
        v, f = r.get("vulnerable_code", ""), r.get("fixed_code", "")
        if not (40 < len(v) < 1500 and 40 < len(f) < 1500):
            return None
        return ("", " " + f.strip(), " " + v.strip())
    return _hf_pairs(n, "hitoshura25/cvefixes", fn)


def _harmfulness_hh(n):
    """hh-rlhf harmless-base as a standalone source (was only a fallback);
    human preferences, fully independent of the PKU pipeline."""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base",
                      split="train", streaming=True)
    tag = "\n\nAssistant:"
    items: List[Item] = []
    for row in ds:
        ch, rj = row.get("chosen", ""), row.get("rejected", "")
        if tag not in ch or tag not in rj:
            continue
        stem = ch[:ch.rfind(tag) + len(tag)]
        if not rj.startswith(stem):
            continue
        good, bad = ch[len(stem):], rj[len(stem):]
        if good.strip() and bad.strip() and good != bad:
            items.append((stem, good, bad))
        if len(items) >= n:
            break
    return items


def _truthfulness_halu(n):
    def fn(r):
        q = r.get("question", "")
        g, b = r.get("right_answer", ""), r.get("hallucinated_answer", "")
        return (f"Q: {q}\nA:", " " + g.strip(), " " + b.strip()) if q else None
    return _hf_pairs(n, "pminervini/HaluEval", fn, config="qa", split="data")


def _truthfulness_sciq(n):
    def fn(r):
        q = r.get("question", "")
        g, b = r.get("correct_answer", ""), r.get("distractor1", "")
        return (f"Q: {q}\nA:", " " + g.strip(), " " + b.strip()) if q else None
    return _hf_pairs(n, "allenai/sciq", fn)


def _bias_stereoset(n):
    """Intersentence StereoSet: preferred = anti-stereotype continuation."""
    from datasets import load_dataset
    ds = load_dataset("McGill-NLP/stereoset", "intersentence",
                      split="validation")
    # The gold_label ClassLabel hides at a different depth depending on
    # whether the loader served the original script schema or the parquet
    # export; canonical order as the last resort.
    feat = ds.features["sentences"]
    names = ["anti-stereotype", "stereotype", "unrelated"]
    for get in (lambda: feat.feature["gold_label"].names,
                lambda: feat["gold_label"].feature.names,
                lambda: feat["gold_label"][0].names):
        try:
            names = get()
            break
        except Exception:                                       # noqa: BLE001
            continue
    items: List[Item] = []
    for row in ds:
        sents = row["sentences"]
        by = {names[l]: s for s, l in zip(sents["sentence"],
                                          sents["gold_label"])}
        g, b = by.get("anti-stereotype"), by.get("stereotype")
        if g and b:
            items.append((row["context"], " " + g.strip(), " " + b.strip()))
        if len(items) >= n:
            break
    return items


def _bias_crows(n):
    def fn(r):
        return ("", " " + r.get("sent_less", "").strip(),
                " " + r.get("sent_more", "").strip())
    return _hf_pairs(n, "quantiles/crows_pairs", fn, split="test")


def _morality_stories(n):
    def fn(r):
        ctx = " ".join(x for x in (r.get("norm"), r.get("situation"),
                                   r.get("intention")) if x)
        g, b = r.get("moral_action", ""), r.get("immoral_action", "")
        return (ctx, " " + g.strip(), " " + b.strip()) if ctx else None
    return _hf_pairs(n, "demelin/moral_stories", fn, config="full")


def _morality_ethics(n):
    """ETHICS utilitarianism ranked pairs. Field semantics differ across
    mirrors; sign is irrelevant for the geometry battery."""
    def fn(r):
        g = r.get("baseline") or r.get("scenario") or ""
        b = r.get("less_pleasant") or r.get("activity") or ""
        return ("", " " + g.strip(), " " + b.strip())
    for repo, cfg in (("hendrycks/ethics", "utilitarianism"),
                      ("lighteval/hendrycks_ethics", "utilitarianism")):
        try:
            items = _hf_pairs(n, repo, fn, config=cfg)
            if items:
                return items
        except Exception as exc:                                # noqa: BLE001
            print(f"    {repo} failed ({str(exc)[:60]})")
    return []


def _toxicity_paradetox(n):
    def fn(r):
        return ("", " " + r.get("en_neutral_comment", "").strip(),
                " " + r.get("en_toxic_comment", "").strip())
    return _hf_pairs(n, "s-nlp/paradetox", fn)


def _sycophancy(n):
    """Perez et al. model-written sycophancy (EleutherAI mirror); preferred =
    the answer NOT matching the stated user opinion. No independent second
    source exists in pair format (verified 2026-08-18) — single-source caveat
    stays attached to any sycophancy-specific claim."""
    def fn(r):
        q = r.get("question", "")
        g = r.get("answer_not_matching_behavior", "")
        b = r.get("answer_matching_behavior", "")
        return (q, g, b) if q else None
    return _hf_pairs(n, "EleutherAI/sycophancy", fn,
                     config="sycophancy_on_nlp_survey", split="validation")


SOURCES = {
    "security": lambda n, seed: _security(n),
    "harmfulness": lambda n, seed: _harmfulness(n),
    "truthfulness": lambda n, seed: _truthfulness(n),
    "dialect": _dialect,
    "spelling": _spelling,
    "grammar": _grammar,
    "security_sven": lambda n, seed: _security_sven(n),
    "security_cve": lambda n, seed: _security_cve(n),
    "harmfulness_hh": lambda n, seed: _harmfulness_hh(n),
    "truthfulness_halu": lambda n, seed: _truthfulness_halu(n),
    "truthfulness_sciq": lambda n, seed: _truthfulness_sciq(n),
    "bias_stereoset": lambda n, seed: _bias_stereoset(n),
    "bias_crows": lambda n, seed: _bias_crows(n),
    "morality_stories": lambda n, seed: _morality_stories(n),
    "morality_ethics": lambda n, seed: _morality_ethics(n),
    "toxicity_paradetox": lambda n, seed: _toxicity_paradetox(n),
    "sycophancy": lambda n, seed: _sycophancy(n),
}

# Category map for the battery: same-category sources fitted independently;
# high within-category direction alignment = the category is real, low = we
# were measuring a dataset.
CATEGORIES = {
    "security": ["security", "security_sven", "security_cve"],
    "harmfulness": ["harmfulness", "harmfulness_hh"],
    "truthfulness": ["truthfulness", "truthfulness_halu", "truthfulness_sciq"],
    "bias": ["bias_stereoset", "bias_crows"],
    "morality": ["morality_stories", "morality_ethics"],
    "toxicity": ["toxicity_paradetox"],
    "sycophancy": ["sycophancy"],
}

# `dialect` and `spelling` bracket the tilt end. `grammar` is NOT a tilt anchor:
# it is the lexical ceiling, the shape a contrast takes when every item differs in
# the same token pair. Reading it as a tilt is the error this comment exists to
# prevent.
TILT_ANCHORS = ("dialect", "spelling")
LEXICAL_ANCHORS = ("grammar",)
ANCHORS = TILT_ANCHORS + LEXICAL_ANCHORS


def load_items(source: str, n: int, seed: int = 0) -> List[Item]:
    if source not in SOURCES:
        raise SystemExit(f"no preference source '{source}'; "
                         f"have {sorted(SOURCES)}")
    items = SOURCES[source](n, seed)
    if not items:
        raise SystemExit(f"source '{source}' yielded no items")
    if len(items) < n:
        print(f"    {source}: only {len(items)}/{n} items available")
    return items


def split_items(items: List[Item], train_frac: float = 0.5, seed: int = 0):
    """Disjoint halves. Every direction is fit on train and read on test — the
    circularity that would otherwise make coherence a measure of overfitting."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * train_frac)
    return ([items[i] for i in idx[:cut]], [items[i] for i in idx[cut:]])


def swap_labels(items: List[Item], seed: int = 0) -> Tuple[List[Item], List[int]]:
    """Flip preferred/dispreferred on a random half.

    The null for every shape statistic. The text is untouched, so per-item
    gradient magnitudes are preserved and only their agreement is destroyed: the
    mean direction must collapse and coherence must fall to its chance value. A
    coherence estimate without this control is a measure of how many items you
    ran.
    """
    rng = random.Random(seed)
    out, signs = [], []
    for p, g, b in items:
        if rng.random() < 0.5:
            out.append((p, b, g))
            signs.append(-1)
        else:
            out.append((p, g, b))
            signs.append(1)
    return out, signs


def wrap_chat(tok, items: List[Item], enable_thinking: bool = False):
    """Score instruct models through their own template.

    A raw continuation scored against a chat-tuned model lands where the model
    expects a turn header, which understates exactly the preference post-training
    installed — and the ladder comparison is about post-training. Qwen-style
    templates open a reasoning block on the generation prompt, so it is closed
    first when the template accepts the argument.
    """
    if not getattr(tok, "chat_template", None):
        return items, False
    wrapped = []
    for prompt, good, bad in items:
        msg = [{"role": "user", "content": prompt}]
        text = None
        for kw in ({"enable_thinking": enable_thinking}, {}):
            try:
                text = tok.apply_chat_template(msg, tokenize=False,
                                               add_generation_prompt=True, **kw)
                break
            except Exception:                                   # noqa: BLE001
                continue
        wrapped.append((text if text is not None else prompt, good, bad))
    return wrapped, True


def maybe_chat(tok, items: List[Item], mode: str) -> Tuple[List[Item], bool]:
    """mode: never | auto | always. `auto` templates only models that have one,
    which is what makes the base/Instruct ladder runnable from one command."""
    if mode == "never":
        return items, False
    return wrap_chat(tok, items)


def describe(items: List[Item], tok=None) -> Dict:
    """Length asymmetry is worth carrying: the secure side of the security pair
    is 43% longer on average, so a raw-sum margin is partly a length preference.
    Every margin is reported per-token for that reason; this records the size of
    the effect being controlled for."""
    out: Dict = {"n": len(items)}
    if tok is None:
        return out
    lg = [len(tok(g, add_special_tokens=False)["input_ids"]) for _, g, _ in items]
    lb = [len(tok(b, add_special_tokens=False)["input_ids"]) for _, _, b in items]
    lp = [len(tok(p, add_special_tokens=True)["input_ids"]) for p, _, _ in items]
    mean = lambda v: sum(v) / max(len(v), 1)
    out.update({"mean_prompt_tokens": mean(lp), "mean_good_tokens": mean(lg),
                "mean_bad_tokens": mean(lb),
                "length_asymmetry": (mean(lg) - mean(lb)) / max(mean(lb), 1e-9)})
    return out
