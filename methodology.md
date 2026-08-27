# Methodology

Working skeleton for the full methodology write-up. Each section below has its
governing question and the material it needs to draw on already committed to
this repo; the prose itself is the next job. Sections marked **[NEW ANALYSIS]**
require data work, not just writing, and should not be started until their
section's question is stated precisely enough to point the analysis at it.

---

## 1. The Estimand

State plainly, before any results: what quantity is being estimated, and over
what population.

- Outcome: `Next_Point_Won` — does the focal player win the point immediately
  following a high-leverage point.
- Treatment: winning a high-leverage point (`High_Leverage` /
  `_BP` / `_TB` == 1 and `Point_Won` == 1). State this as a **design choice**,
  not an incidental detail: the control pool is *mixed* — it contains both
  points where the focal player lost the high-leverage point and ordinary,
  non-high-leverage points, pooled together. Say why (keeps ATP and WTA, BP
  and TB, on a consistent comparison group) and say what it costs (the
  contrast isn't "won HLP vs lost HLP" — that sharper contrast is future work,
  see §7).
- Estimand: the ATE of that treatment on `Next_Point_Won`, adjusted for
  `Focal_Ranking`, `Rolling_Win_Pct`, `Streak_k4`, `CUSUM`.
- This section is where the grade critique's core gap gets closed: define the
  object before reporting numbers about it.

## 2. Two Questions, Not One

The falsification design answers two logically separate questions, and
conflating them was the specific gap the grade critique flagged. State them
as two questions up front, then structure the results section to answer them
in order:

1. **Is it real?** — Is there a genuine causal effect of winning a
   high-leverage point on the next point? Answered by the ATE + its
   confidence interval (§4, §5).
2. **Is it structural or psychological?** — *Given* a real effect exists, is
   it explained by the serving-transition mechanism, or by psychological
   momentum? Answered by the break-point/tiebreak contrast: BP creates a
   serving-structure advantage for the winner, TB does not (serve alternates
   every point/two points, neutralising the mechanism). Momentum should show
   up in *both* if it's real; the mechanical account predicts it only in BP.

Write the contrast explicitly as a contrast: τ_BP vs τ_TB, not as two
independent findings that happen to be reported together.

## 3. Mechanism: Why the Contrast Is Valid

The BP/TB contrast is only informative if the mechanism claim is true. This
is now empirically verified, not asserted — state it as settled:

- Every break point immediately followed by a game/set change was won by the
  *receiver* (1,183/1,183 ATP, 887/887 WTA), confirming the `Pts` score
  format assumption the BP detection logic depends on.
- The BP winner serves the next point in 100.0000% of cases with a valid next
  point (3,117/3,117 ATP, 1,938/1,938 WTA) — not a pattern, a structural
  certainty of the server-alternates-every-game rule: server holds → same
  server continues; returner converts → game ends → returner serves the new
  game.
- Frame this as a proven mechanism, not a provisional one — it can anchor the
  argument without hedging.

## 4. Inference: What Was Wrong and What Replaced It

- The bug: `cates.std() / sqrt(n)` is the dispersion of the forest's per-unit
  CATE *predictions*, not the sampling variance of the ATE, and it treats
  serially-dependent points within a match as independent observations.
- The fix: a bootstrap that resamples whole matches (not points), refitting
  the estimator on each resample and taking the spread of resulting ATEs as
  the SE. Implemented in `scripts/bootstrap_ate.py`.
- The estimator split: CausalForestDML where treatment is well-powered
  (combined, break-point cells); LinearDML for the two tiebreak cells (sparse
  treatment) and the four rank-only robustness specs (single-control
  propensity fragility) — state the rule as a principle, with its evidence:
  a cv-fold sweep showed the forest's tiebreak point estimate flipping sign
  under grouped cross-fitting (wta_tb: +0.039 / -0.104 / -0.015 across
  cv=2/3/5) while LinearDML reproduced the forest's own outlier-trimmed
  median to within ~5% in every one of six affected cells, with a clean
  bootstrap distribution each time.
- BLB reframing: econml's native bootstrap-of-little-bags inference targets
  the forest's own parameter uncertainty, not ATE sampling variance under
  match clustering. Report it, if at all, as "econml's native forest
  inference, targeting a different quantity" — not as a rung on the same
  ladder as the clustered bootstrap CI, which is the number to report as the
  interval.

### 4a. Two further bugs, found in a later audit pass, both fixed and refit

Two construction errors were found auditing the pipeline after §7a's SGP work
shipped. Neither threatens the qualitative findings — both were re-verified by
a full refit before being reported here, not assumed safe.

- **CUSUM's neutral-fill bug.** `features.py`'s CUSUM computed
  `series.shift(1).fillna(0)` *before* `.expanding().mean()`, so a fabricated
  "loss" at the start of every match was permanently baked into the running
  mean for the rest of that match — unlike `Rolling_Win_Pct`, whose neutral
  fill is applied *after* its rolling calculation. Confirmed on real data: for
  ATP matches where the focal player's win rate was 48–52% (should show
  CUSUM ≈ 0 if the measure means what it's documented to mean), mean CUSUM was
  +1.94 pre-fix. Fix: move the neutral fill to after the subtraction
  (`(shifted - expanding_mean).fillna(0)`), so the fabricated value never
  enters the running mean's denominator — only the single first-point
  deviation, genuinely undefined with no prior data, resolves to 0. Verified
  post-fix: the same near-50% check now gives −0.12 (a ~93% reduction, and
  correctly near zero rather than strongly positive); a constant-outcome
  sequence (all wins or all losses) now produces flat zero CUSUM throughout,
  as it should for a perfectly consistent performer with no deviation from
  their own average; and a no-look-ahead check (flip only a sequence's last
  point, confirm every earlier CUSUM value is byte-identical) passed, so the
  fix does not reintroduce leakage while closing the bias.
- **Control-pool contamination.** Every spec's control pool was defined as
  `Treatment==0`, i.e. "everything else" — which silently included other
  high-leverage types' *won* points as if they were neutral baseline. Since
  each type has its own large, non-zero effect, this wasn't neutral: e.g. the
  BP spec's control pool contained ~10% won server-game points (effect ≈
  −0.14), pulling BP's baseline down and inflating BP's apparent effect.
  Quantified directly on raw means (ATP): the "clean" ordinary-points-only
  baseline is 0.5100; BP's contaminated control sat at 0.4975 (−1.25pp),
  Combined's at 0.4971 (−1.28pp), TB's at 0.5029 (−0.71pp, also inflating,
  i.e. pulling TB away from null), SGP's at 0.5158 (+0.58pp, inflating SGP's
  magnitude negative). Fix: both `bootstrap_ate.py`'s `build_spec()` and
  `model.py`'s equivalent construction (`run_causal_forest`, `run_logistic`,
  `check_vif` — these duplicate `build_spec`'s logic and must be kept in sync,
  per the module's own docstring) now exclude every *other* treatment's
  won-points from a spec's control, leaving only points untouched by any
  high-leverage flag.
- **Both fixes were folded into one refit**, not applied and reported
  separately, since the control-pool fix alone would have forced a rerun
  regardless of CUSUM's status. Full before/after table in §5's source CSVs
  (`outputs/ate_results.csv`, `outputs/ate_results_sgp.csv`,
  `outputs/ate_results_robustness.csv`); all headline numbers throughout this
  document, the blog, and the README reflect the post-fix pipeline.
- **BP's move, decomposed** (isolated by refitting each fix in isolation,
  holding the other at its old/buggy value — not inferred from the combined
  move): ATP's ~−0.0158 move is control-pool-dominated (−0.0110 from the pool
  fix alone vs. −0.0019 from CUSUM alone, ~85%/12% of the net, small positive
  synergy). WTA's ~−0.0217 move has **both fixes contributing comparably**
  (−0.0154 pool, −0.0130 CUSUM), with a mild sub-additive interaction (the
  isolated effects sum to more than the net move) — not two large offsetting
  moves cancelling to a small net, but not a single dominant cause either.
  This asymmetry between tours is itself worth remembering: "the contamination
  fix moved BP" is the accurate one-line summary for ATP; for WTA it's "both
  fixes moved it, roughly equally."
- **What was checked before trusting the refit, not assumed:** both nulls
  (TB, both tours) retain comparable CI width post-fix (ATP: 0.076→0.0756;
  WTA: 0.1428→0.1649, ~15% wider but still comfortably spanning zero) and
  both still clearly span zero — the null's character is unchanged, not just
  its point estimate. SGP's sign held on both tours under the identical fixed
  pipeline (ATP −0.1394, WTA −0.0725 — WTA moved slightly *more* negative,
  ATP barely moved). Feature importance was the one genuinely open question —
  whether CUSUM's dominance was partly a points-elapsed artifact riding on the
  bias — and it was checked with no prior on the answer: CUSUM still dominates
  post-fix (54.4% ATP / 57.5% WTA, vs. ~55%/59% pre-fix), so the "accumulated
  drift" reading of feature importance survives on the numbers as they stand,
  not because it was assumed to.

### 4b. Structural refit: DML cross-fitting default changed from cv=2 to cv=5

- **The change.** The pipeline's DML cross-fitting fold count, previously
  defaulting to 2, now defaults to 5 — `bootstrap_ate.py`'s `fit_ate`,
  `bootstrap_spec`, and CLI default, and `model.py`'s `CausalForestDML`
  instantiation in `run_causal_forest`. This is pipeline-wide, not
  forest-only: every spec, forest and LinearDML alike, takes a `cv`
  parameter, unlike the earlier cv=10 sensitivity sweep (§4a-era), which was
  deliberately scoped to the six well-powered forest specs only.
- **Why 5, not just "higher."** cv=2 is the minimum fold count that supports
  cross-fitting at all (one fold trains the nuisance models, the other
  scores the treatment effect on held-out data) and sits at the
  high-nuisance-bias end of the bias/variance tradeoff cross-fitting
  exists to manage: with only two folds, each nuisance model is trained on
  roughly half the data, which can leave residual overfitting bias in the
  Y- and T-models that DML's own theory assumes cross-fitting removes.
  cv=5 is a standard default in the double-ML literature and in `econml`'s
  and `scikit-learn`'s own conventions, balancing that bias reduction
  against the variance cost of training nuisance models on smaller folds —
  it is not chosen because more folds is unconditionally better (the
  sparse cells are exactly the counterexample: §4a's original cv=10 sweep,
  and this section's own bracket below, both treat very high fold counts on
  thin cells as a finite-sample-instability risk, not a virtue).
- **What moved.** Small shifts on every spec, no sign changes on any
  significant cell, both nulls stayed null:

  | | cv=2 | cv=5 (new default) |
  |---|---|---|
  | ATP Combined | 0.0844 [0.0688, 0.1128] | 0.0888 [0.0701, 0.1135] |
  | WTA Combined | 0.0478 [0.0196, 0.0785] | 0.0550 [0.0205, 0.0805] |
  | ATP BP | 0.1312 [0.1069, 0.1588] | 0.1362 [0.1071, 0.1602] |
  | WTA BP | 0.0646 [0.0273, 0.0948] | 0.0671 [0.0270, 0.0943] |
  | ATP TB | −0.0066 [−0.0285, 0.0471] | +0.0072 [−0.0232, 0.0437] |
  | WTA TB | −0.0091 [−0.0911, 0.0738] | +0.0024 [−0.0853, 0.0657] |
  | ATP SGP | −0.1394 [−0.1502, −0.1111] | −0.1398 [−0.1526, −0.1130] |
  | WTA SGP | −0.0725 [−0.0932, −0.0276] | −0.0581 [−0.0923, −0.0318] |

  Both tiebreak point estimates flip sign again between cv=2 and cv=5 (as
  they did between the pre-fix and post-§4a-fix pipelines) — both were
  already indistinguishable from zero and remain so; a near-zero estimate
  crossing zero between specifications is expected noise around a genuine
  null, not evidence of anything. Feature importance: CUSUM still dominates
  (53.3% ATP / 51.8% WTA, vs. 54.4%/57.5% at cv=2) — WTA's margin narrowed
  more than ATP's (~6pp vs. ~1pp) but the ranking order (CUSUM ≫ Ranking ≈
  Rolling_Win_Pct ≫ Streak_k4) is unchanged.
- **The cv=2/5/10 robustness bracket (forest specs only, B=199 each,
  current pipeline).** Resolves §4a/§7a.7 item 7, which had cut a cv=10
  check run on the pre-§4a pipeline rather than let it read as current. All
  three fold counts now run on the identical, current (fixed-CUSUM,
  clean-control-pool) pipeline:

  | | cv=2 | cv=5 | cv=10 |
  |---|---|---|---|
  | ATP Combined | 0.0844 | 0.0888 | 0.0876 |
  | WTA Combined | 0.0478 | 0.0550 | 0.0536 |
  | ATP BP | 0.1312 | 0.1362 | 0.1349 |
  | WTA BP | 0.0646 | 0.0671 | 0.0682 |
  | ATP SGP | −0.1394 | −0.1398 | −0.1408 |
  | WTA SGP | −0.0725 | −0.0581 | −0.0578 |

  No sign changes anywhere across the bracket; every CI at every fold count
  overlaps substantially with its neighbours. One honest asymmetry, not
  smoothed over: ATP SGP is essentially flat across all three fold counts
  (−0.1394/−0.1398/−0.1408, a tight range), while WTA SGP shows real
  point-estimate movement between cv=2 (−0.0725) and cv=5/10
  (−0.0581/−0.0578) — cv=5 and cv=10 agree closely with each other but not
  with cv=2. The CI-based conclusion (comfortably negative, clear of zero
  at every fold count: [−0.0932,−0.0276], [−0.0923,−0.0318],
  [−0.0930,−0.0320]) holds throughout regardless, but "stable across
  cv=2/5/10" is a claim about the sign and the CI, not a claim that WTA
  SGP's point estimate is fold-count-invariant — it isn't, quite.

## 5. Results

- Headline table: all 10 specs, ATE + match-clustered bootstrap 95% CI
  (percentile, not ±1.96·SE — the WTA forest specs' bootstrap distributions
  are visibly skewed, so report the percentile bounds directly). Source:
  `outputs/ate_results.csv`, `outputs/ate_results_robustness.csv`.
- Feature importance: state the correct interpretation before showing the
  chart — importance measures *effect heterogeneity* (how much the estimated
  effect varies across the forest's splits), not effect size. "Winning Streak
  scores ~2%" does not by itself refute a hot-hand story; it says streak
  status doesn't much change *how large* the break-point effect is for a
  given player, which is a narrower claim than "streaks don't predict
  anything" (that's addressed separately by the independence tests in an
  earlier section).

## 6. The WTA Weighting

State explicitly, with justification — this is an epistemic claim about your
own evidence, not a footnote:

- The falsification claim rests primarily on the ATP arm; the WTA arm
  corroborates but is weaker, and should be presented as directional rather
  than co-equal.
- The evidence for this weighting is not just the smaller WTA tiebreak sample
  (209 treated / 128 matches) — it's that the WTA tiebreak forest estimate
  was demonstrably unstable under arbitrary, methodologically-irrelevant
  configuration choices (cv-fold count) before the estimator switch. LinearDML
  resolved the *point estimate*, but the fact that it needed resolving is
  itself information about how much weight this cell can bear.
- This is the sentence that belongs here, not in the blog: the blog states
  the factual flag (smaller sample); this document argues the weighting.

## 7. Robustness

- **Rank-only (single-control) specs**: already run and stable — all four
  LinearDML re-runs land within ~5% of the forest's own outlier-trimmed
  median, with clean bootstrap distributions. Evidence in
  `diagnostics/inference_bootstrap/`.
- **Retirement/walkover points**: disclosed limitation, not a fix — the
  charting matches file has no score column, so retirement points remain in
  the point-level data (only the rankings lookup filters them). State this
  plainly; note it's unlikely to bias the BP/TB contrast specifically (no
  reason retirement points would concentrate differently across leverage
  types) but say that's a plausibility argument, not a test.
- **Rival non-psychological explanations and the matched-comparison test** —
  see §7a below. Provisional draft; Test 1 run, Test 2 reuses existing results.

### 7a. Robustness — Two Rivals, Two Tests: The Matched-Comparison Design

> **STATUS: PROVISIONAL DRAFT, TEST 1 NOW RUN.** This section replaces an earlier
> flat-vs-rising leverage-gradient design (see git history) that was diagnosed as
> invalid: serve-transition only fires at break points, which are high-leverage by
> construction, so a rising effect-by-leverage gradient is *also* consistent with
> pure serve-transition and cannot discriminate it from discouragement. The
> replacement is a matched-comparison design (§7a.3) plus the existing tiebreak null
> (§7a.4), each assigned to a different rival. Test 1 has been run to full precision
> (match-clustered bootstrap, B=199); its point estimates and the mechanism check
> that motivates the design are reported below. Open: the Morris measure remains
> unbuilt (deferred — see §7a.7 item 1, the crude leverage match was checked and
> found close); the assumption that either rival mechanism *persists in tiebreaks*
> is inference, not a tested claim in either source paper (§7a.7 items 2–3); the WTA
> tiebreak cell remains power-limited (§7a.7 item 4). All headline figures below
> reflect a control-pool and CUSUM-construction fix found in a later audit pass and
> folded into a full refit — see §4a for what changed, by how much, and what was
> checked before trusting it. §4b is a second, later structural refit: the
> pipeline's DML cross-fitting default changed from cv=2 to cv=5, with a
> cv=2/5/10 robustness bracket re-run on the current pipeline (resolving
> §7a.7 item 7, previously a cut, unrun check). All BP/SGP figures below are
> the cv=5 values.

#### 7a.1 Three rivals, not one

The falsification design of §2–§3 establishes that apparent break-point momentum is
large where serve transfers and undetectable where serve alternates. That contrast
rules out simple hot-hand momentum as the *sole* explanation. It does not, on its
own, rule out two further rivals — both non-hot-hand, both capable of producing the
same BP-present/TB-null pattern the original design tests for.

**Rival 2: discouragement (strategic momentum).** Gauriot and Page (2019) exploit
near-the-line Hawk-Eye calls as a quasi-experimental shock to point outcomes and
find that, for male professional players, winning a point raises the probability of
winning the next point by roughly 7.2 percentage points, rising sharply with stakes
(≈2.5 pp at 0–0, ≈10.6 pp at 30–30/deuce). They interpret this as a *rational
effort* effect: the trailing player's continuation value drops, so they rationally
ease off — momentum without any psychological or physiological state. A referee can
grant the entire serve-transition story and still argue some or all of the measured
break-point effect is this: the player who just lost a break point faces a degraded
continuation value and eases off on the next point, independent of who serves it.

**Rival 3: belief-updating.** Descamps, Ke, & Page (2022), "How success breeds
success," *Quantitative Economics*, 13(1), 355–385 (DOI: 10.3982/QE1679), is a lab
study documenting that success in one round of a competitive task raises a
participant's belief about their own ability, which in turn raises effort and
performance in the following round — a rational, Bayesian channel distinct from
both the "hot hand" cognitive bias (which §2–§3's original tests address) and from
discouragement (which is about the *loser's* falling continuation value, not the
winner's rising self-belief). Applied to tennis: winning a point could raise a
player's belief about their own current form, boosting their next-point win
probability independent of who serves it.

Both rivals predict exactly the BP-present pattern the original falsification design
looks for, and — this is the reason a matched-comparison design is needed at all —
neither is ruled out by that design alone. Distinguishing them from serve-transition,
and from each other, is part of establishing the spine's causal claim, not optional
robustness.

#### 7a.2 Two tests, one primary against both rivals, one corroborating

- **Test 1 (§7a.3, new): the matched-leverage, opposite-serve-direction comparison.**
  Primary evidence against **both** discouragement and belief-updating. Both rivals
  predict the same sign at a high-leverage point the treated player wins, regardless
  of which player serves next: discouragement because the *loser's* continuation
  value drops (benefiting the winner next point, whoever serves it); belief-updating
  because the *winner's* self-assessment rises (benefiting the winner next point,
  whoever serves it). Serve-transition is a structural effect: its sign should track
  who serves next, which — as shown below — runs in *opposite* directions at break
  points and at the matched comparison group. Opposite signs across the two groups
  is evidence against both rivals being the primary driver; same-signed effects at
  both would have left either or both live.
- **Test 2 (§7a.4, existing): the tiebreak null.** Corroborating evidence,
  specifically for belief-updating. Belief-updating is about the winner's own
  updated self-assessment; it should not care who serves the next point, so it
  should survive into tiebreaks even though serve alternates there and the
  serve-transition channel is neutralised. A null effect in tiebreaks corroborates
  Test 1's finding on belief-updating in a setting where serve-transition is fully
  neutralised rather than merely reversed; it is not the primary evidence, since
  Test 1 already contradicts belief-updating's sign prediction directly (§7a.5).

Test 1 alone already bears on both rivals; Test 2 adds a second, independent,
serve-neutral (rather than serve-reversed) check on the weaker of the two —
see §7a.5 for why neither test, alone, proves either rival's contribution is zero.

#### 7a.3 Test 1: The matched-leverage, opposite-serve-direction comparison

**The comparison group.** Server game points — the server one point from holding,
or already at advantage after deuce — are the mirror image of break points in the
score string: `Pts ∈ {40-0, 40-15, 40-30, AD-40}` versus break point's
`{0-40, 15-40, 30-40, 40-AD}`. Engineered as `High_Leverage_SGP` in `clean.py`,
symmetric to the existing `High_Leverage_BP` logic; deuce (`40-40`) and tiebreak
points (numeric `Pts` strings under the `Gm1==6 & Gm2==6` condition) are excluded by
construction, since neither matches either string set.

**Mechanism check — this is not the "no transition" group it was first described
as.** The working assumption going in was that winning a server game point holds
serve with no transition, making it serve-neutral at matched leverage. Checked
directly the same way §3 verified the break-point mechanism (does the point winner
serve the immediately following point, using `Svr`/`PtWinner` on consecutive points
within a match):

| | winner serves next point |
|---|---|
| Break point (`High_Leverage_BP`) | **100.0000%** (3,117/3,117 ATP, 1,938/1,938 WTA) |
| Server game point (`High_Leverage_SGP`) | **0.0000%** (7,244/7,244 ATP, 2,969/2,969 WTA) |

Server game points are not serve-neutral. They are the *exact structural mirror* of
break points: at a break point the point winner always serves next (receiver
converts → game ends → by strict service rotation the just-converted receiver serves
the new game; server saves → game continues → same server continues, trivially).
At a server game point the point winner *never* serves next (server wins → game
ends, holds — but rotation still hands the next game to the other player regardless
of who won it; receiver wins → game merely continues, and the original server keeps
serving mid-game). This is a deterministic fact of the service-alternation rule, not
a pattern — same certainty class as §3's 100% figure. The design is therefore not
"leverage matched, serve-transfer removed" as first framed; it is "leverage matched,
serve-transfer *reversed*." That is a cleaner test than the one originally
specified: it predicts opposite-signed effects, not merely the absence of an effect
at the comparison group.

**Leverage comparability (crude match, checked; Morris measure deferred — see
§7a.7 item 1).** Comparing the won-treatment populations
(`High_Leverage_{BP,SGP}==1 & Point_Won==1`) on the two match/set-context axes
identified as carrying real spread (§7a.3's original leverage-spread pre-check,
game-margin and set number):

| axis | ATP BP (n=1,497) | ATP SGP (n=3,645) | WTA BP (n=1,001) | WTA SGP (n=1,464) |
|---|---|---|---|---|
| Game-margin \|Gm1−Gm2\|, median [IQR] | 1 [0, 2] | 1 [0, 2] | 1 [0, 2] | 1 [0, 2] |
| Set number, distribution | spans all sets, no concentration | spans all sets, no concentration | spans all sets, no concentration | spans all sets, no concentration |

Match/set-context leverage is close to identical across the two groups on both
tours. The one axis that differs is within-game score state itself — BP is skewed
toward the closer-to-deuce states (30-40/40-AD ≈ 66% of the population, both tours)
while SGP spreads more evenly across its four states, including 40-0/AD-40 at the
edges — but that axis was already flagged as genuinely coarse for BP alone (§7a.3's
original pre-check, 4 discrete values by construction) and the crude match on the
axes that do carry spread is close enough that a Morris-importance re-weighting is
not expected to change the qualitative comparison. Deferring the principled measure
to the shipped version per the explicit reconnaissance/ship split (§7a.7 item 1);
this result is reconnaissance-grade.

**Power.** Both cells are well above the break-point cells they're compared
against — 3,645 ATP / 1,464 WTA treated versus BP's 1,497 / 1,001 — so this is not
an underpowered comparison the way the tiebreak cells are.

**Why a mirror, not a neutral control.** Break points are one instance of a broader
question: what happens to next-point odds after a high-leverage win? Tennis doesn't
offer an equally-high-stakes point where the outcome doesn't touch who serves next —
every such point either transfers serve or confirms it. The nearest substitute is
the server's own game point, defined above. It isn't neutral — the point winner
serves next 0% of the time at a server game point versus 100% at a break point, both
verified directly against consecutive-point serve records (§7a.3's mechanism check)
— and that reversal is what makes it useful: if the break-point effect is the serve,
the server-game-point effect should run the other way.

**The result.** It does. Match-clustered bootstrap ATE (CausalForestDML, same
estimator and controls as the BP/combined specs — well-powered, stays on the forest
per the estimator-split rule in §4), B=199, cv=5, on the clean-control-pool,
fixed-CUSUM pipeline (§4a, §4b): winning a break point raises next-point win
probability by +0.1362 in the ATP (95% CI [0.1071, 0.1602]) and +0.0671 in the
WTA ([0.0270, 0.0943]). Winning a server game point lowers it: −0.1398 in the ATP
([−0.1526, −0.1130]) and −0.0581 in the WTA ([−0.0923, −0.0318]). All four
intervals sit entirely on one side of zero, and the sign flips with the serve.
Stable across a cv=2/5/10 robustness bracket (§4b) — no sign changes at any fold
count, though WTA SGP's point estimate moves somewhat between cv=2 and cv=5/10
while staying comfortably clear of zero throughout.

**Why the reversal is robust.** The reversal is robust in a way no single effect
estimate is. The two treatments share one defining feature: in each, it is the focal
player who wins the point. Any mechanism operating through winning — discouragement,
belief-updating, momentum — must therefore assign both effects the same sign,
whatever else differs between them. A sign reversal cannot be produced by a
winning-based channel at all; it can only come from what the two treatments do not
share, which is the direction of the serve. Two secondary checks close the remaining
gaps. The treatments draw on the same control pool, so any baseline artefact common
to both shifts the estimates together rather than in opposite directions. And the
two flags are built as a literal mirror of the point-score string — break point is
`Pts ∈ {0-40, 15-40, 30-40, 40-AD}`, server game point is the same four states with
the server/receiver halves swapped, `Pts ∈ {40-0, 40-15, 40-30, AD-40}` — so the
contrast is not built on obviously mismatched stakes; a formal Morris (1977)
importance match, deferred here, would confirm the leverage alignment directly
rather than resting on the mirror construction alone.

**No magnitude claim is made.** The BP and SGP CIs overlap substantially in
magnitude on both tours — ATP BP [0.1071, 0.1602] against SGP's magnitude
[0.1130, 0.1526] (fully nested), WTA BP [0.0270, 0.0943] against SGP's
magnitude [0.0318, 0.0923] (also fully nested) — so
the effects are not statistically distinguishable in size, and nothing here should
be read as "mirror-image" or "equal and opposite" in magnitude, on either tour. That
would be a stronger claim than the data support and isn't needed: the sign flip
alone, with all four CIs clear of zero, is what does the discriminating work, and it
stands on its own regardless of how the magnitudes compare.

**Cross-fitting sensitivity (cv=10): retable, not run on the current pipeline.**
An earlier cv=10 sweep (six well-powered specs, B=199) found the SGP sign held
at a different DML fold count on the pre-§4a pipeline. That check no longer
describes a pipeline that exists — it ran on the contaminated control pools and
biased CUSUM this document's §4a fix replaced, so its numbers are not a valid
claim about the current, published estimates. Rather than let a stale check
read as a current one with a "the old numbers still happen to fall inside the
new CIs, so it's probably fine" hedge, it is cut here. Cross-fitting-fold
robustness on the current pipeline is an open item — see §7a.7 item 7 — not an
established one; nothing above should be read as implying it.

#### 7a.4 Test 2: The tiebreak null, corroborating belief-updating

The existing tiebreak result (§5: ATP +0.0072, CI crosses zero; WTA +0.0024, CI
crosses zero) was originally read as evidence against hot-hand momentum generally.
It corroborates the belief-updating result Test 1 already establishes (§7a.5),
in a setting where the serve-transition channel is fully neutralised (serve
alternates) rather than merely reversed (as at server game points) — a cleaner but
much less powerful isolation of the belief-updating question specifically, since
belief-updating (Descamps, Ke, & Page 2022) is about the winner's own updated
self-assessment and has no structural reason to depend on who serves the following
point.

**Two honesty flags that must survive into the final text, not get smoothed away:**

1. **The "persists in tiebreaks" claim is inference, not citation — and no longer
   load-bearing on its own.** Descamps, Ke, & Page (2022) is a lab study of
   belief-updating after success in a competitive task; it is silent on tennis, on
   tiebreaks, and on point-serve structure specifically. That their mechanism
   *would* persist when serve alternates is a plausible extrapolation from what
   belief-updating is (a property of the winner, not of who serves next), not a
   claim their paper makes or tests. Treat this as reasoning about what their
   mechanism implies, not as their finding. It matters less now than it did before
   Test 1 was run: Test 1's negative SGP sign already contradicts belief-updating's
   own positive prediction directly (§7a.5), so this flag qualifies a corroborating
   check, not the primary evidence. The same caveat, already on record, applies to
   discouragement: whether Gauriot and Page's continuation-value account predicts
   persistence in tiebreaks specifically has not been checked against their actual
   formalism (carried forward from the original open item; still unverified — see
   §7a.7 item 2) — likewise no longer load-bearing, since Test 1 is discouragement's
   primary evidence and doesn't depend on this claim.
2. **The tiebreak null is power-limited, not a proof of absence.** WTA's 209
   treated tiebreak points (§6) make this evidence-against-belief-updating-at-current-
   power, not evidence that belief-updating is structurally excluded. The same
   caveat that already governs §6's WTA weighting applies here.

#### 7a.5 Reading the two tests together

Test 1 alone already bears on both rivals, because both predict the same sign
(positive, winner-favouring) regardless of who serves next, and the observed sign at
server game points is negative on both tours. That is evidence against **both**
discouragement and belief-updating being the *primary* driver of the break-point
effect. It is not evidence that either channel's contribution is exactly zero: a
small positive contribution from either mechanism could be present and simply
swamped by a dominant negative serve-transition effect at server game points,
since Test 1 estimates the net effect, not each channel separately. **The correct
claim is "not the primary driver," not "ruled out" or "absent."**

Test 2 (the tiebreak null) adds a second, weaker, but independent check specific to
belief-updating, in a setting (serve fully neutralised rather than reversed) that
isolates the belief-updating question more cleanly than Test 1 does — at the cost of
the severe power limitation already on record for the tiebreak cells (§6, §7a.7 item
4). It corroborates Test 1's belief-updating result; it does not rescue the case if
Test 1 hadn't found what it found, and it says nothing about discouragement, which
operates on abundant regular-play data Test 2 doesn't touch.

**Joint conclusion.** Test 1's opposite-signed, all-CIs-clear-of-zero pattern at
matched leverage is what serve-transition predicts and what no stakes-based or
belief-based rival predicts under a same-sign-regardless-of-serve assumption —
that is the primary result, sufficient on its own to identify serve-transition as
the *primary* driver and both rivals as not primary. The tiebreak null corroborates
the belief-updating piece specifically, in an independent, serve-neutral (not
merely serve-reversed) setting, at low power. Neither result is a claim that
discouragement or belief-updating contribute nothing at all — only that neither is
the primary, detectable driver of this effect, on this data, at current power.

This does not retire either rival's citation from the document — Gauriot and Page's
and Descamps, Ke, and Page's mechanisms remain real, documented phenomena in
competitive settings generally. What the joint test licenses is narrower: that they
are not, at current power and on this data, the primary driver of *this*
break-point effect specifically.

#### 7a.6 Sex-asymmetry cross-reference

Serve-transition is sex-neutral (serve rotates identically on both tours);
discouragement and belief-updating are plausibly male-specific, by analogy to a
documented, male-specific set-level winner effect in tennis: Page and Coates (2017)
find that among equally-matched players (restricting to matches where the first set
went to a long tiebreak, >20 points, so the set winner is quasi-random with respect
to ranking), the winner of that tiebreak takes the second set ~60% of the time among
men, versus ~51% among women (~53% to the loser) — effectively absent. They propose
an androgen-mediated mechanism (testosterone elevating confidence and performance in
the following set). **This reference supports two things here and not a third: that
a winner effect in tennis is documented, and that it is male-specific. It does not
support, and this document does not claim, that our point-level mechanical account
overturns or subsumes their set-level physiological one — different level of
analysis, different mechanism, no refutation implied.** [Citation verified against
project notebook extract, not the primary source; see §7a.7 item 5.]

The WTA arm of Test 1 shows the same opposite-signed pattern as ATP (WTA BP +0.0671
vs WTA SGP -0.0581), which supports reading the WTA break-point effect as
sex-neutral structural momentum rather than a fragile artifact — consistent with,
not proof of, serve-transition operating identically on both tours. This is flagged
as a question the test bears on, **not** as a contribution claim: the WTA cells
remain the smaller-power arm (§6), and the flattering interpretation is not to be
assumed beyond what the point estimates show.

#### 7a.7 Open dependencies and unverified claims

1. **Morris measure deferred to the shipped version, not built.** Test 1 (§7a.3)
   ran on the crude score-state/game-margin/set-number match, found close enough on
   the axes that carry real spread (game-margin, set number) that a Morris-importance
   re-weighting is not expected to change the qualitative result. This is
   reconnaissance-grade, flagged as such — the win-probability-swing measure is still
   not built and remains required before journal submission. **Citation, verified
   against the primary source (not a notebook extract):** Klaassen, C. A. J., &
   Magnus, J. R. (2001). Are points in tennis independent and identically
   distributed? Evidence from a dynamic binary panel data model. *Journal of the
   American Statistical Association*, 96(454), 500–509. p.12 reproduces Morris's
   definition verbatim and states "This definition was first suggested by Morris
   (1977)," full cite: Morris, C. (1977), "The Most Important Points in Tennis," in
   *Optimal Strategies in Sport*, eds. S.P. Ladany and R.E. Machol, North-Holland,
   131–140. Point importance = the difference in the player's probability of
   winning the game/set conditional on winning versus losing the current point —
   the win-probability swing. Klaassen & Magnus operationalise it from two fixed
   pre-match point-win probabilities, which is the direct template for building the
   measure when it ships. Bonus, p.17: they show the result is robust to
   substituting a break-point dummy for full importance and still rejecting IID —
   independent support for leaning on break-point/server-game-point status as a
   leverage-adjacent variable in Test 1 (§7a.3).
2. **UNVERIFIED mechanistic claim (carried forward):** the assertion that
   discouragement *persists in tiebreaks* has **not** been checked against Gauriot
   and Page's actual continuation-value model. It is a plausible hypothesis, not an
   established joint, and must not be load-bearing — Test 1, not the tiebreak null,
   is the primary evidence against discouragement (§7a.2), which is a further reason
   this item doesn't block the joint conclusion in §7a.5.
3. **UNVERIFIED mechanistic claim (new), downgraded — no longer load-bearing
   alone.** The assertion that belief-updating (Descamps, Ke, & Page 2022) *persists
   in tiebreaks* is this document's inference from what belief-updating is (a
   property of the winner, independent of who serves next), not a claim tested in
   their paper, which is a lab study silent on tennis and on serve structure. Must
   not be load-bearing beyond "a plausible extrapolation this document is relying
   on" for the tiebreak corroboration specifically — but this item no longer gates
   the belief-updating conclusion generally, since Test 1's negative SGP sign
   already contradicts belief-updating's own positive prediction directly, without
   relying on any tiebreak-persistence assumption (§7a.5).
4. **WTA tiebreak null is power-limited (carried forward from §6).** 209 treated
   points. Evidence-against-belief-updating-at-current-power, not proof of exclusion.
5. **Citation sourced (Page & Coates):** Page, L., & Coates, J. M. (2017). Winner
   and loser effects in human competitions. Evidence from equally matched tennis
   players. *Evolution and Human Behavior*, 38(4), 530–535. Correctly scoped to what
   it supports (a documented, male-specific winner effect), not claimed to refute
   their set-level mechanism — see §7a.6. **Provenance note:** the Page & Coates and
   Gauriot & Page citations in this section are verified against project notebook
   extracts, not the primary sources directly — adequate for this draft; a
   primary-source spot-check on the 60/40, 51/53, >20-point-tiebreak threshold
   (Page & Coates) and the 7.2pp/2.5pp/10.6pp figures (Gauriot & Page) is owed before
   journal submission, same as any other load-bearing quote.
6. **Descamps, Ke, & Page (2022) citation:** verified directly against the paper
   (held in the project folder), not a notebook extract or secondary source —
   Quantitative Economics, 13(1), 355–385, DOI 10.3982/QE1679. Higher provenance
   standard than items above; no primary-source spot-check owed for the citation
   itself, only for item 3's inference about tiebreak persistence.
7. **Resolved: cross-fitting-fold robustness on the current pipeline.** The
   item this used to flag (an earlier cv=10 sweep that ran on the pre-§4a
   pipeline, cut rather than left reading as current) is closed. §4b re-runs
   the check properly: a cv=2/5/10 bracket on the current (fixed-CUSUM,
   clean-control-pool) pipeline, all six well-powered specs, B=199 each. No
   sign changes at any fold count. One thing the bracket surfaced that
   wasn't visible from a single cv=10 point estimate: WTA SGP's point
   estimate is not fold-count-invariant (−0.0725 at cv=2 vs.
   −0.0581/−0.0578 at cv=5/10) even though its sign and CI-clear-of-zero
   status are — see §4b for the exact numbers and the honest framing of
   what "stable" does and doesn't claim there.

## 8. Limitations

- Dataset skews toward elite matches (charting project coverage); can't speak
  to lower-tier players.
- No physiological confounds measured (fatigue, injury short of retirement).
- Treatment definition is a design choice (§1) — a within-HLP,
  serve-conditioned specification (comparing won-HLP directly against
  lost-HLP, conditioning on who serves next) would sharpen the psychological
  vs structural distinction further; noted as the natural next specification,
  not attempted here.
- Single season, single year (2023) — multi-season extension is
  robustness-stage work for a journal version, not required for this
  document's claims.

## 9. Register

Paper-standard throughout; no hyperbole. (The blog is allowed a punchier
register — that boundary is deliberate, see §6 — this document is not.)

---

*Status: §1–6 and §8–9 are skeleton, ready to fill from material already
committed. §7a (rival explanations and the matched-comparison design) is a
full provisional draft, with Test 1 now run. The earlier flat-vs-rising
leverage-gradient design (option 1/option 2 treatment-definition fork) has
been retired — it couldn't discriminate serve-transition from discouragement,
since serve-transition also predicts a rising gradient once you condition on
break points being high-leverage by construction. Its replacement, the
matched-comparison test (§7a.3: break points vs. server game points, the
mirror-image score states), has been run to full precision and reproduces
exactly on a fresh process: opposite-signed effects on both tours (ATP break
point +0.1362 vs. server game point −0.1398; WTA +0.0671 vs. −0.0581, on the
clean-control-pool, fixed-CUSUM, cv=5 pipeline — see §4a, §4b), all four CIs
clear of zero, and stable across a cv=2/5/10 robustness bracket (§4b) with no
sign changes anywhere. The claim rests on the sign only, not the magnitude —
the BP/SGP CIs overlap substantially in size on both tours (e.g. ATP
[0.1071,0.1602] vs. [0.1130,0.1526] in magnitude), so there is no mirror-image
or equal-and-opposite claim here, only that both signs are clean and opposite.
That sign flip tracks a deterministic mechanism check — the point winner
serves next 100.0000% of the time at break points and 0.0000% of the time at
server game points, both tours. Test 1 alone is already primary evidence
against both discouragement and belief-updating (§7a.2, §7a.5), since both
predict the same winner-favouring sign regardless of who serves next, and
none of the four CIs comes close to that prediction. The existing tiebreak
null (§7a.4, reassigned as corroborating evidence specifically for
belief-updating, via Descamps, Ke, & Page 2022) adds a second, independent,
serve-neutral check at low power — it does not carry the argument alone the
way it would have under the retired design. What remains, listed at §7a.7:
the Morris/Klaassen-Magnus importance measure is deferred to the shipped
version (the crude match was checked and found close on the axes that carry
real spread — game-margin, set number — so this is flagged
reconnaissance-grade, not blocking); Test 1, corroborated by Test 2, rules
out discouragement and belief-updating as the *primary* driver, not as
present-in-any-degree — a small effect from either could be swamped by a
dominant serve-transition effect, so "not primary" is the claim, not
"absent"; the claim that either mechanism persists in tiebreaks specifically
is this document's inference, not a tested claim in either source paper; and
the WTA tiebreak cell remains power-limited (209 treated). The Page & Coates
(2017) and Descamps, Ke, & Page (2022) citations
are sourced — the latter verified directly against the paper, the former
against project notebook extracts with a primary-source spot-check still owed
before journal submission. Nothing in §7a should be treated as analysis-ready
until the remaining open items clear.*
