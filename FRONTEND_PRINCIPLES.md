# FRONTEND_PRINCIPLES — the founding rules of the MRI-Lab interface

Draft v1, 2026-09-01 (from the mock round + user notes). This is the document the
frontend is built FROM; the mocks were the search, this is the crystallization.
Every rule exists because the blueprint's ideas will live in this interface —
if the interface misstates them, the program misstates them.

Status: FOR DISCUSSION — §8 lists the decisions still open between us.

---

## 0. What this thing is

Not a dashboard. An **instrument bench with rooms**: each room answers one
question about one subject, with measured readings placed on measured surfaces,
under the program's evidence discipline. The webapp is the blueprint's public
body — U1 (process control), U2 (forensics), U3 (alignment doctrine), U4
(transport calculus) each get walls here, and nothing gets a wall before its
cell lands.

Audience: professionals (researchers, fine-tuners, releasers). Tone:
lab-notebook declarative. The program's own vocabulary IS the brand —
MEASURED / VOID BY GATE / NOT LICENSED / WITHDRAWN are interface states, not
internal jargon to hide.

---

## 1. The spine: three doors, one subject model  ← (your "not only checkpoints")

Everything in the app hangs off a SUBJECT. There are exactly three kinds:

  DOOR A — CHECKPOINT   "I have weights."       repo@revision or local path.
                        → the biography rooms (readiness, fragility, economics,
                          post-training fingerprint) + preflight.
  DOOR B — RECIPE       "I have an architecture   arch config + dataset intents.
                        and datasets."            → what the map licenses TODAY:
                          descriptive corpus crash-profiles, placement rules
                          (rule-shaped data after the crossing; safety after ITS
                          crossing), epoch advisories, the sensor plan for the
                          run. NO architecture-chooser (no results back one) —
                          the door configures instruments, it does not promise
                          outcomes.
  DOOR C — LIVE RUN     "I am training now."     checkpoint stream / dir watch.
                        → the observatory: clocks, smoothed gauges, CI gates.
                          Advisory only until a DEMO-CL union lands.

One subject model underneath: a RECIPE becomes a RUN becomes CHECKPOINTS — the
doors are entry points into one lifecycle, and the UI keeps that thread visible
(a checkpoint page names its run; a run page names its recipe).

## 2. Rooms, not a dashboard  ← (your "dashboards should not be all in one page")

**One room = one question.** A room may be dense; the app may not be one page.

  GATE          which subject, which door?         (landing; airy)
  SCAN          what IS this checkpoint?           (summary: verdict chips +
                                                    state portrait + doors into
                                                    the four rooms below)
  READINESS     what does teaching become here?    (surface, crossings, payload
                                                    classes, safety crossing)
  FRAGILITY     what breaks it, per channel?       (quantization rank; crash
                                                    probes; provenance of its
                                                    curvature)
  ECONOMICS     what does behaviour cost here?     (exchange table, persistence,
                                                    collateral)
  LINEAGE       what did its recipe do to it?      (post-training fingerprint,
                                                    stage ladder, norm clock)
  PREFLIGHT     should I fine-tune it like this?   (wizard: intent → probe plan
                                                    → jobs → 2-D verdict)
  OBSERVATORY   how is my run developing?          (door C; clocks + gates)
  REGISTRY      what does the program claim?       (claims browser; retraction
                                                    wall as a first-class view)
  ARENA         what do we bet against the field?  (standing predictions)
  QUEUE         what is the machine doing?         (jobs, logs, GPU state)

Navigation: persistent subject bar (switch subject anywhere), room tabs in a
fixed order (the lifecycle order above), ⌘K jump. A room NEVER embeds another
room's full content — it links with a one-line measured teaser.

## 3. Density gradient  ← (your "bench best, but too dense for landing")

Depth in the app = density on the screen.

  GATE        one action, one sentence, one quiet figure. Air.
  SCAN        the bench header: chips + portrait + four teasers. Medium.
  ROOMS       full bench density — tables, figures, evidence chrome. Dense.
  RAW         the row JSON itself. Maximum.

Corollary (your "doesn't use the full space well"): rooms are **workbench
layouts** — full-viewport grids that ADD columns at 1440/1920px (the figure
gains a side panel; tables gain columns) instead of stretching whitespace.
Line length stays ≤ 70ch for prose, but instruments and tables claim the width.

## 4. Every figure is an instrument  ← (your "cool but not explanatory")

A figure that needs the author to explain it is rejected. Contract for ANY
visualization (the state portrait included):

  a. CAPTION: one line stating what is encoded ("tremor = held-out loss;
     spokes = unseen-rule gain; hatch = isotropic curvature").
  b. TOUCH: hovering/clicking any visual element names its quantity, its
     CURRENT value with units, and links the claim that licenses it.
  c. "HOW TO READ THIS": every figure has an annotated overlay toggle (?) —
     the same figure with callout arrows. First visit auto-offers it once.
  d. LEGEND = GAUGES: the legend entries ARE the live readouts (the number
     sits next to the swatch), so learning the encoding is reading the data.
  e. NOTHING DECORATIVE ON DATA: every visual channel maps to a measured
     quantity; if a channel is aesthetic-only it goes to the page chrome,
     never onto the figure.
  f. The state portrait (glyph) uses ONE encoding app-wide — learn it once
     in SCAN, read it everywhere (small in headers, full in OBSERVATORY).

## 5. The evidence chrome (the program's discipline as UI law)

  - Every rendered number can open its provenance in ≤ 2 interactions:
    value → claim (class · n · bounds, verbatim) → source row JSON + doc §.
  - REFUSALS ARE RENDERED. A scope-guard denial is a visible "NOT LICENSED:
    <reason>" block, never an empty space. The refusal is the product's
    honesty made tangible — it is a feature, styled as one.
  - RETRACTIONS ARE FURNITURE. The registry's WITHDRAWN/KILLED entries render
    as a revision-history wall; reports carry their relevant retractions
    inline. Misses as loud as hits, structurally.
  - RANKS AND CROSSINGS, NEVER FORECASTS. Copy rules: "rank 6/6 among measured
    pairs", "past the crossing", "matched arms say…" — banned phrasings:
    "will", "expect X±Y" cross-family, any magnitude extrapolation (L1).
  - CI-CLEAR DISCIPLINE IN PIXELS: points whose CI spans zero render as
    no-verdict (hollow/hatched), and the crossing band is drawn as a band.
  - LIVE GAUGES ARE SMOOTHED. No raw single-probe value is ever displayed as
    a trigger-like signal (DEMO-CL: n=24 swings ±0.8) — series show the
    smoothing window, and there are NO knob controls anywhere until a
    DEMO-CL union lands.
  - GAPS NAME THEIR UNLOCK. "Not measured" always says which cell/run unlocks
    it ("→ Cell Q-bis", "→ run 3 probes, ~27 min"). A gap is an invitation
    with a price tag, not a dead end.

## 6. Tool physics (the "I despise heavy frontends" law)

  - BUDGETS: first meaningful paint < 1s warm, < 2.5s cold; ≤ 75 KB
    transferred per room (excluding fonts); JS ≤ 30 KB per room; zero
    layout shift after first paint; no spinners for < 300ms operations.
  - Report/room views render from the report JSON as HTML that works with
    JS disabled; interactivity is progressive (islands), never a boot screen.
  - STATE LIVES IN THE URL: /m/{repo@rev}/{room}, /run/{id}, /claims/{id},
    /reports/{id} — every view addressable, shareable, refreshable. Reports
    are permalinks to immutable JSON.
  - KEYBOARD-FIRST: ⌘K subject/jump, [ ] scrub, j/k rows, ? figure overlay,
    g then r/f/e room hops. Hints visible in chrome (bench style).
  - Long work is honest: job states come from the engine's table; logs
    stream; ETAs quote MEASURED timings ("~13 min at 1.4b — measured") or
    say nothing.
  - Reduced-motion honored everywhere; the app is fully usable static.

## 7. Character, budgeted  ← (your "more interesting, but professional")

Identity = **the working artifacts of a measurement lab**, taken seriously:
graph-paper figure grounds, tolerance rings, stamp marks, part-number blocks,
bracket controls [ SCAN ], kbd chrome, verdict decals, revision tables. The
character comes from the subject's own world — never from decoration.

  - ONE SIGNATURE MOMENT PER ROOM. Gate: the inked state portrait breathing
    on graph paper. Fragility: the quench hatch. Registry: the stamp wall.
    Everything around the moment stays quiet.
  - MOTION ENCODES MEASUREMENT OR DOESN'T EXIST. Scrubbing morphs the
    portrait (data), state changes tick (events); zero ambient animation.
  - TYPE: IBM Plex Mono (data/values) + IBM Plex Sans (labels/prose) — chosen
    for the instrument lineage, tabular numerals everywhere digits align.
  - PALETTE: lab-paper light-first (paper/ink/teal + semantic ok/warn/block),
    full considered dark theme (not an inversion). One accent; semantic
    colors are reserved and never decorative.
  - BANNED (the AI-default look, explicitly): purple-blue gradient heroes,
    glassmorphism, emoji as section markers, floating-screenshot heroes,
    rounded-2xl card fields, acid-green-on-black hacker costume, CRT/scanline
    skeuomorphism, boot animations, Inter-because-default.
  - Wit is allowed in microcopy ("No sedation required."), one per room, never
    on refusals or retractions.

## 8. DECIDED 2026-09-01 (user)

  1. THEME: lab-paper LIGHT-FIRST, with a full considered dark theme.
  2. ROOM NAMES: FUNCTIONAL register — Load, Overview, Readiness, Fragility,
     Economics, Lineage, Preflight, Run Monitor, Registry, Arena, Queue.
     (The bench-metaphor words survive only inside copy, never as nav.)
  3. THE PORTRAIT IS CANON, frozen: tremor = held-out loss · spokes =
     unseen-rule gain (readiness) · warm hatch = isotropic curvature ·
     dashed tolerance ring = ‖W‖²/N · red shards = anti-generalization ·
     crossing pulse at sign change. Any change to this encoding is a
     versioned decision recorded here, like an instrument revision.
  4. STACK: server-rendered HTML from report JSON + vanilla-JS islands.
  5. RECIPE DOOR v1 INCLUDES the GPU corpus crash-profile runner from day one
     (dp_profile + erosion_arms jobs through the queue — backend already
     supports both; the door renders profiles as DESCRIPTIVE, per the DP v1.1
     withdrawal).
  6. MATERIAL: CONSOLE (bordered panels, chip nav, kbd chrome) over the print
     specimen treatment (round-3 A/B). Print survives only as salvage: the
     stamp, FIG-style captions where a figure earns one.
  7. THE LIFE KIT (user: "more life, same language") — warmth and micro-detail
     from the lab's own world, applied system-wide:
       · manila folder-tab section headers (lab folders; a warm second
         MATERIAL tint, never a semantic color)
       · verdict chips filled with a soft wash of their semantic color
       · drafting registration marks on figure-panel corners + a ruler strip
         along the figure edge + a center crosshair on the portrait
       · the brand is the portrait: mini-glyph + wordmark in the subject bar
       · discovered delight: hovering a legend-gauge row highlights that
         channel on the portrait (everything else dims); buttons depress 1px
       · still zero ambient animation; all of it holds in both themes.

## 8b. What "interesting" means here (position, adopted with §7)

Interest in a tool has four durable sources, in this order:

  1. THE PHENOMENON, NOT THE PAINT. The most interesting object on any page
     is that the subject is real and alive: a model quenching, a crossing you
     can scrub across, a probe landing and a verdict ticking over. Design's
     job is to put the viewer's hand on the instrument — presence of
     measurement beats decoration of it, always. (Most AI-built frontends are
     boring because there is nothing under them; we have a phenomenon.)
  2. OPINION IN THE DETAILS. Character is the accumulation of choices nobody
     else would make: refusals styled as features, retractions as furniture,
     verdict decals, VOID-BY-GATE as interface vocabulary, the stamp. The web
     feels identical not for lack of ornament but for lack of conviction —
     our epistemics are the one thing no other tool can copy, so they carry
     the identity.
  3. DISCOVERED DELIGHT, NEVER PERFORMED. The earned moment per room (§7)
     rewards attention instead of demanding it: the figure highlighting a
     channel when you hover its legend gauge, the overlay teaching you once,
     the stamp's 1.2° tilt. If a delight plays before the user does anything,
     cut it.
  4. TEXTURE OVER MOTION. The lost "style" of the web is material, not
     animation: graph-paper grounds, double-rule mastheads, figure numbers
     and captions, tabular numerals, revision tables — print-grade care that
     makes the page feel like a made object. Our material world is the
     instrument/datasheet lineage; that reads as character to professionals
     because it signals care, not branding.

  THE GOVERNING TENSION: first-visit wow decays; hundredth-visit affection
  compounds. We design for the hundredth visit (speed, keyboard flow,
  density, a voice you'd miss) and let the first visit's wow be the
  phenomenon itself — the portrait is allowed to be spectacular BECAUSE it
  is honest.

## 9. Acceptance test for any future page

Before a page ships, it must pass:
  [ ] Which single question does this room answer? (one sentence)
  [ ] Can a stranger read every figure unaided? (contract §4 a–d present)
  [ ] Does every number reach its claim in ≤2 interactions?
  [ ] Are refusals/gaps/retractions rendered where relevant?
  [ ] Budgets met (§6)? Keyboard path exists? Reduced-motion clean?
  [ ] Does it read as OURS with the signature moment — and nothing else loud?
