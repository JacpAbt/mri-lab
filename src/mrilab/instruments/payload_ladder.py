# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: experiments/regimes/payload_ladder.py
# source sha256: f6a63d140de48c00f395ccee2ea9032ca63106f15a9c8b82caa00e7a035c4b9d
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""CELL O (INSTRUMENT_PREREG_2026_08.md + PANEL_PREREG amendment): the
payload ladder — anchors of graded semantic depth, every rung shipping its
own seen/unseen family split, a FORMAT TWIN (same surface, different
content-rule) and, where constructed, a CONTENT TWIN (same rule, different
surface). Item pools are FROZEN here; the sweep only reads them.

  L0  LEXICAL          dialect (existing; pairs[0:16] seen / [16:24] unseen)
                       format twin: spelling variants in the dialect
                       narrative frames (same surface, different rule)
  L1  MORPHOSYNTACTIC  a constructed conditional inflection: vowel-initial
                       stems pluralise with "-im", consonant-initial with
                       "-ol". Unseen members are FRESH STEMS, so
                       generalization requires the phonological CLASS
                       mapping, not stem memory.
  L2  PROPOSITIONAL    "prefer the continuation that names its source".
                       QA surface primary; CONTENT TWIN = the same rule in
                       a code-comment surface; format twin = QA_DIALECT.
  L3  SEMANTIC         truth-preference across independent fact sources:
                       demos from sciq, evaluated within-source (held-out
                       sciq) and across constructs (HaluEval, TruthfulQA).
                       Format twin = QA_DIALECT. The pressure/sycophancy
                       veto is carried by the existing honesty_battery
                       cells on the big models, not re-measured here.

Family membership: L0/QA_DIALECT = the substitution pair; L1 = the stem;
L2 = the (question, claim, source) triple; L3 = the fact source. Carriers
and phrasings rotate WITHIN a member and never define membership.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from mrilab.instruments import prefs
from mrilab.instruments.icl_readiness import items_for as dialect_items_for

Item = Tuple[str, str, str]

# ---------------------------------------------------------------- L1 MORPH
V_SEEN = ["apple", "orange", "engine", "island", "anchor", "essay",
          "onion", "oven"]
C_SEEN = ["table", "garden", "bottle", "candle", "mirror", "pillow",
          "basket", "ladder"]
V_UNSEEN = ["umbrella", "envelope", "insult", "eagle", "exit", "album"]
C_UNSEEN = ["carpet", "window", "hammer", "ribbon", "saddle", "lantern"]
PLURAL_FRAMES = [
    "They counted six", "The crate contained several",
    "She photographed many", "The list mentioned four",
    "He stacked up twelve", "Nobody wanted those five",
]
SUF_V, SUF_C = "im", "ol"


def morph_items(which: str, n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    stems = ([(s, True) for s in (V_SEEN if which == "seen" else V_UNSEEN)]
             + [(s, False) for s in (C_SEEN if which == "seen" else C_UNSEEN)])
    combos = [(f, s, v) for f in PLURAL_FRAMES for s, v in stems]
    rng.shuffle(combos)
    out = []
    for f, s, v in combos[:n]:
        good = f" {s}{SUF_V if v else SUF_C}"
        bad = f" {s}{SUF_C if v else SUF_V}"
        out.append((f, good, bad))
    return out


def morph_demos(k: int, seed: int = 0) -> List[Item]:
    """One demo per SEEN stem, frames rotating — every family member shown."""
    rng = random.Random(seed)
    stems = [(s, True) for s in V_SEEN] + [(s, False) for s in C_SEEN]
    rng.shuffle(stems)
    out = []
    for i, (s, v) in enumerate(stems[:k]):
        f = PLURAL_FRAMES[i % len(PLURAL_FRAMES)]
        out.append((f, f" {s}{SUF_V if v else SUF_C}",
                    f" {s}{SUF_C if v else SUF_V}"))
    return out


# --------------------------------------------------------------- L2 ATTRIB
# 24 (question, claim, source) triples; 16 seen, 8 unseen. Claims banal and
# true-ish; sources generic documents, never real organisations.
TRIPLES = [
    ("Where does the sun rise", "the sun rises in the east", "the almanac"),
    ("How many legs do spiders have", "spiders have eight legs", "the field guide"),
    ("What do bees collect", "bees collect nectar and pollen", "the survey"),
    ("When does water boil", "water boils at one hundred degrees", "the manual"),
    ("Where do penguins mostly live", "most penguins live in the southern hemisphere", "the atlas"),
    ("What is bread made from", "bread is made mostly from flour and water", "the handbook"),
    ("How long is a standard year", "a standard year lasts three hundred sixty-five days", "the registry"),
    ("What melts ice", "heat melts ice into water", "the bulletin"),
    ("What do caterpillars become", "caterpillars become butterflies or moths", "the field notes"),
    ("Which metal rusts", "iron rusts when exposed to moisture", "the workshop log"),
    ("What pulls objects downward", "gravity pulls objects toward the ground", "the primer"),
    ("Where is salt found", "salt is abundant in seawater", "the ledger"),
    ("What do plants need for growth", "plants need light, water and nutrients", "the garden manual"),
    ("How do fish breathe", "fish breathe through their gills", "the aquarium guide"),
    ("What season follows spring", "summer follows spring", "the calendar notes"),
    ("What is thunder", "thunder is the sound produced by lightning", "the weather log"),
    # ---- unseen from here (8) ----
    ("What do cows mainly eat", "cows mainly eat grass and hay", "the farm records"),
    ("Where does rain come from", "rain falls from condensing clouds", "the station log"),
    ("What are icebergs made of", "icebergs are made of frozen fresh water", "the expedition notes"),
    ("How do birds fly", "birds fly by flapping their wings for lift", "the aviary guide"),
    ("What makes sound travel", "sound travels as vibrations through air", "the acoustics primer"),
    ("What is honey made by", "honey is made by bees from nectar", "the apiary records"),
    ("Which way do compasses point", "compasses point toward magnetic north", "the navigation manual"),
    ("What do volcanoes release", "volcanoes release lava and ash", "the geology bulletin"),
]
Q_PHRASINGS = ["Q: {q}?\nA:", "Question: {q}?\nAnswer:",
               "Q: {q}, exactly?\nA:", "Someone asked: {q}?\nReply:"]


def _attrib(which: str, n: int, seed: int, surface: str) -> List[Item]:
    rng = random.Random(seed)
    triples = TRIPLES[:16] if which == "seen" else TRIPLES[16:]
    combos = [(p, t) for p in Q_PHRASINGS for t in triples]
    rng.shuffle(combos)
    out = []
    for p, (q, claim, src) in combos[:n]:
        if surface == "qa":
            prompt = p.format(q=q)
            good = f" According to {src}, {claim}."
            bad = f" {claim[0].upper()}{claim[1:]}."
        else:  # code-comment surface (the CONTENT twin)
            prompt = f"# question: {q.lower()}?\n# answer:"
            good = f" according to {src}, {claim}"
            bad = f" {claim}"
        out.append((prompt, good, bad))
    return out


def attrib_qa_items(which: str, n: int, seed: int = 0) -> List[Item]:
    return _attrib(which, n, seed, "qa")


def attrib_code_items(which: str, n: int, seed: int = 0) -> List[Item]:
    return _attrib(which, n, seed, "code")


def attrib_demos(k: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    triples = list(TRIPLES[:16])
    rng.shuffle(triples)
    return [(Q_PHRASINGS[i % 4].format(q=q), f" According to {src}, {claim}.",
             f" {claim[0].upper()}{claim[1:]}.")
            for i, (q, claim, src) in enumerate(triples[:k])]


# --------------------------------------- shared format twin: QA_DIALECT
QA_PROMPTS = ["Q: How does the letter end?\nA:",
              "Q: What does the diary entry say?\nA:",
              "Q: What did the witness add?\nA:",
              "Q: How does the report close?\nA:"]


def qa_dialect_items(which: str, n: int, seed: int = 0) -> List[Item]:
    """Dialect substitution pairs re-shelled into the QA surface."""
    rng = random.Random(seed + 7)
    base = dialect_items_for(which, n, seed)
    return [(rng.choice(QA_PROMPTS), g, b) for _, g, b in base]


# --------------------------------------- L0 format twin: SPELL_IN_FRAMES
SPELL_FRAMES = ["the letter mentioned the {0} twice and nothing else",
                "nothing in the file explained the {0}",
                "the whole final page was about the {0}",
                "she wrote at length about the {0} of it"]
LETTER_PROMPTS = ["The last paragraph of the letter read:",
                  "His diary entry for that day says:",
                  "The witness statement continues:",
                  "She wrote back the same evening:"]


def spell_in_frames_items(which: str, n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    nouns = prefs._SPELLING_NOUNS
    half = len(nouns) // 2
    pool = nouns[:half] if which == "seen" else nouns[half:]
    combos = [(p, f, w) for p in LETTER_PROMPTS for f in SPELL_FRAMES
              for w in pool]
    rng.shuffle(combos)
    return [(p, " " + f.format(br), " " + f.format(us))
            for p, f, (br, us) in combos[:n]]


# ------------------------------------------------------------- L3 TRUTH
def truth_items(source: str, lo: int, hi: int) -> List[Item]:
    """Deterministic slice of a truth source (loaders iterate in fixed
    order). sciq[0:16] is the demo set and never appears in any eval."""
    fn = {"sciq": prefs._truthfulness_sciq,
          "halu": prefs._truthfulness_halu,
          "tqa": prefs._truthfulness}[source]
    return fn(hi)[lo:hi]


def truth_demos(k: int = 16) -> List[Item]:
    return truth_items("sciq", 0, k)


# ------------------------------------------------------ the rung registry
# eval sets: name -> (builder, n). "seen" is the G1 gate set everywhere.
RUNGS: Dict[str, Dict] = {
    "L0_dialect": {
        "demos": lambda k, seed=0: dialect_items_for("seen", k, seed),
        "max_len": 640,
        "evals": {
            "seen": lambda n: dialect_items_for("seen", n, 1234),
            "unseen": lambda n: dialect_items_for("unseen", n, 1234),
            "format_twin": lambda n: spell_in_frames_items("seen", n, 1234),
            "content_twin": lambda n: qa_dialect_items("unseen", n, 1234),
        },
    },
    "L1_morph": {
        "demos": morph_demos,
        "max_len": 640,
        "evals": {
            "seen": lambda n: morph_items("seen", n, 1234),
            "unseen": lambda n: morph_items("unseen", n, 1234),
            "format_twin": lambda n: [
                (f, g, b) for f, g, b in
                [(fr, " " + w[0], " " + w[1]) for fr in PLURAL_FRAMES
                 for w in prefs._DIALECT][:n]],
        },
    },
    "L2_attrib": {
        "demos": attrib_demos,
        "max_len": 896,
        "evals": {
            "seen": lambda n: attrib_qa_items("seen", n, 1234),
            "unseen": lambda n: attrib_qa_items("unseen", n, 1234),
            "format_twin": lambda n: qa_dialect_items("unseen", n, 1234),
            "content_twin": lambda n: attrib_code_items("unseen", n, 1234),
        },
    },
    "L3_truth": {
        "demos": lambda k, seed=0: truth_demos(k),
        "max_len": 1280,
        "evals": {
            "seen": lambda n: truth_items("sciq", 16, 16 + n),
            "unseen": lambda n: truth_items("halu", 0, n),
            "unseen2": lambda n: truth_items("tqa", 0, n),
            "format_twin": lambda n: qa_dialect_items("unseen", n, 1234),
        },
    },
}

# L1's format twin: dialect single-word substitutions inside the PLURAL
# frames — same surface as morph items (frame + one word), different rule.


# ---------------------------------------------------- L1-BIS counter-prior
ADJ_NOUNS_SEEN = [("red", "ball"), ("old", "house"), ("small", "table"),
                  ("green", "door"), ("heavy", "box"), ("bright", "lamp"),
                  ("narrow", "street"), ("quiet", "room"), ("tall", "tree"),
                  ("cold", "river"), ("dusty", "shelf"), ("broken", "clock"),
                  ("yellow", "coat"), ("empty", "bottle"), ("sharp", "knife"),
                  ("wooden", "chair")]
ADJ_NOUNS_UNSEEN = [("blue", "car"), ("young", "dog"), ("wide", "window"),
                    ("warm", "blanket"), ("thin", "rope"), ("dark", "cellar"),
                    ("round", "mirror"), ("soft", "pillow"),
                    ("long", "corridor"), ("clean", "towel"),
                    ("steep", "hill"), ("plain", "envelope")]
ADJ_FRAMES = ["She pointed at", "They walked past", "He finally noticed",
              "The photo showed", "Nobody mentioned", "It stood beside"]

ART_NOUNS_SEEN = ["apple", "orange", "engine", "island", "umbrella",
                  "envelope", "eagle", "oven", "table", "garden", "bottle",
                  "candle", "mirror", "pillow", "basket", "ladder"]
ART_NOUNS_UNSEEN = ["anchor", "essay", "onion", "insult", "exit", "album",
                    "carpet", "window", "hammer", "ribbon", "saddle",
                    "lantern"]
ART_FRAMES = ["On the table she found", "He said it looked like",
              "They had never seen", "The letter arrived with",
              "Somebody left behind", "We were offered"]


def _vowel(w: str) -> bool:
    return w[0].lower() in "aeiou"


def adjorder_items(which: str, n: int, seed: int = 0) -> List[Item]:
    """Counter-prior: PREFER noun-adjective order."""
    rng = random.Random(seed)
    pairs = ADJ_NOUNS_SEEN if which == "seen" else ADJ_NOUNS_UNSEEN
    combos = [(f, a, o) for f in ADJ_FRAMES for a, o in pairs]
    rng.shuffle(combos)
    return [(f, f" the {o} {a}", f" the {a} {o}") for f, a, o in combos[:n]]


def article_items(which: str, n: int, seed: int = 0) -> List[Item]:
    """Counter-prior: PREFER the wrong article."""
    rng = random.Random(seed)
    nouns = ART_NOUNS_SEEN if which == "seen" else ART_NOUNS_UNSEEN
    combos = [(f, w) for f in ART_FRAMES for w in nouns]
    rng.shuffle(combos)
    out = []
    for f, w in combos[:n]:
        right, wrong = (" an ", " a ") if _vowel(w) else (" a ", " an ")
        out.append((f, wrong + w, right + w))     # preferred = WRONG article
    return out


def _cp_demos(items_fn, k: int, seed: int = 0) -> List[Item]:
    return items_fn("seen", k, seed)


RUNGS["L1B_adjorder"] = {
    "demos": lambda k, seed=0: _cp_demos(adjorder_items, k, seed),
    "max_len": 640,
    "evals": {
        "seen": lambda n: adjorder_items("seen", n, 1234),
        "unseen": lambda n: adjorder_items("unseen", n, 1234),
        "format_twin": lambda n: [
            (f, " " + w[0], " " + w[1]) for f in ADJ_FRAMES
            for w in prefs._DIALECT][:n],
    },
}
RUNGS["L1B_article"] = {
    "demos": lambda k, seed=0: _cp_demos(article_items, k, seed),
    "max_len": 640,
    "evals": {
        "seen": lambda n: article_items("seen", n, 1234),
        "unseen": lambda n: article_items("unseen", n, 1234),
        "format_twin": lambda n: [
            (f, " " + w[0], " " + w[1]) for f in ART_FRAMES
            for w in prefs._DIALECT][:n],
    },
}


# ---------------------------------------------------- L1-TER nonce-form test
# The morph anchor (L1) required discriminating two INVENTED word-forms and
# decayed with capability. L1-ter tests whether NONCE-FORM discrimination is
# the operative variable with a second anchor of a different KIND: a
# constructed lexicon of nonce nouns, each assigned an arbitrary nonce
# CLASSIFIER particle (like a counter word), split so unseen members require
# the (nonce-noun -> nonce-particle) association, not a phonological rule.
# Contrast partner already exists: dialect (real words, prior-consistent) at
# the same k, which RISES with capability — so a within-cell dissociation
# (nonce decays, real rises) is the clean read, not an absolute level.
NONCE_NOUNS_SEEN = ["blicket", "dax", "fendle", "gorp", "hemtel", "jimjin",
                    "kraadle", "lonk", "murdle", "nabbin", "poffin",
                    "quannet", "risp", "sundle", "tannet", "wug"]
NONCE_NOUNS_UNSEEN = ["vernit", "yomble", "zatch", "brindle", "chulp",
                      "drant", "flimble", "grock", "hesper", "plimf",
                      "snorl", "tromble"]
NONCE_PARTICLES = ["mo", "ta"]     # two arbitrary classifier particles
NONCE_FRAMES = ["She pointed at the", "He held up the", "They counted one",
                "On the shelf sat a", "Nobody wanted the",
                "The box contained one"]

# EXACTLY BALANCED by construction (caught by the build smoke: a hash-based
# assignment gave 10/16 seen and 9/12 unseen to one particle, so a constant
# "prefer mo" strategy would have scored ~75% on unseen and faked the rule).
# Alternating over the declared list order makes any constant strategy
# chance-level on BOTH splits, which is what the unseen gain must clear.
_NONCE_CLASS = {w: i % 2 for i, w in enumerate(NONCE_NOUNS_SEEN)}
_NONCE_CLASS.update({w: i % 2 for i, w in enumerate(NONCE_NOUNS_UNSEEN)})


def _nonce_class(word: str) -> int:
    """Arbitrary but BALANCED assignment; learnable only by memorising
    members (no phonological or orthographic cue predicts it)."""
    return _NONCE_CLASS[word]


def nonce_items(which: str, n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    nouns = NONCE_NOUNS_SEEN if which == "seen" else NONCE_NOUNS_UNSEEN
    combos = [(f, w) for f in NONCE_FRAMES for w in nouns]
    rng.shuffle(combos)
    out = []
    for f, w in combos[:n]:
        cls = _nonce_class(w)
        good = f" {w} {NONCE_PARTICLES[cls]}"
        bad = f" {w} {NONCE_PARTICLES[1 - cls]}"
        out.append((f, good, bad))
    return out


RUNGS["L1T_nonce"] = {
    "demos": lambda k, seed=0: nonce_items("seen", k, seed),
    "max_len": 640,
    "evals": {
        "seen": lambda n: nonce_items("seen", n, 1234),
        "unseen": lambda n: nonce_items("unseen", n, 1234),
        "format_twin": lambda n: [
            (f, " " + w[0], " " + w[1]) for f in NONCE_FRAMES
            for w in prefs._DIALECT][:n],
    },
}


def register_for_battery() -> None:
    """Expose the constructed rungs as prefs sources so pref_battery can
    measure their lexical shares without editing the frozen prefs module."""
    prefs.SOURCES["morph"] = lambda n, seed: morph_items("seen", n, seed)
    prefs.SOURCES["attrib_qa"] = lambda n, seed: attrib_qa_items("seen", n, seed)
    prefs.SOURCES["attrib_code"] = lambda n, seed: attrib_code_items("seen", n, seed)
    prefs.SOURCES["qa_dialect"] = lambda n, seed: qa_dialect_items("seen", n, seed)
    prefs.SOURCES["spell_frames"] = lambda n, seed: spell_in_frames_items("seen", n, seed)
