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
- **The strategic-momentum objection and the leverage-scaling test** —
  see §7a below. Provisional draft; not yet run.

### 7a. Robustness — The Strategic-Momentum Objection and the Leverage-Scaling Test

> **STATUS: PROVISIONAL DRAFT.** This section rests on three open dependencies
> (proxy result pending, Morris measure not yet built, discouragement-in-tiebreaks
> unverified against the source model) plus one design item to action before the
> proxy runs (the leverage measure needs broadening beyond within-game score state
> — spread check found that axis alone gives only 4 buckets for the treated
> population) plus one resolved-but-flagged item (the Page & Coates citation is now
> sourced and correctly scoped, with a primary-source spot-check still owed before
> journal submission) — all listed at §7a.7. The treatment-definition fork is
> resolved (option 1 primary, option 2 as positive control — see §7a.3). This is a
> specification of a test not yet run, not a report of a result. Do not read any
> sentence below as claiming the effect is flat
> across leverage — that is the empirical question this section exists to
> answer.

#### 7a.1 The objection

The falsification design of §2–§3 establishes that apparent break-point momentum is
large where serve transfers and undetectable where serve alternates. That contrast
rules out a *psychological* reading of the effect. It does not, on its own, rule out
a second non-psychological rival, and that rival is the more dangerous of the two
because it is already established in the same empirical setting.

Gauriot and Page (2019) exploit near-the-line Hawk-Eye calls as a quasi-experimental
shock to point outcomes and find that, for male professional players, winning a point
raises the probability of winning the next point by roughly 7.2 percentage points.
Crucially, the effect is not uniform: it is near zero at the start of a game (≈2.5 pp
at 0–0) and rises sharply in symmetric high-stakes states (≈10.6 pp at 30–30 / deuce).
They interpret this as *strategic* momentum — a discouragement effect. When the score
becomes asymmetric, the trailing player faces a lower continuation value: the effort
required to recover is high and the immediate reward low, so a rational competitor
reduces effort, while the leader's relative incentive to press rises. The momentum is
real but arises from asymmetric rational effort, not from a psychological or
physiological state.

This is a rival explanation of *our positive result*, not of our null. A referee can
grant the entire serve-transition story and still argue that some — or all — of the
measured break-point effect is discouragement: the player who has just lost a break
point faces a degraded continuation value and rationally eases off on the following
point, independent of who serves it. Serve-transition and discouragement both predict
exactly the pattern we observe. Distinguishing them is therefore not optional; it is
part of establishing the spine's causal claim.

#### 7a.2 The discriminating prediction

The two mechanisms make *opposite* predictions about how the effect varies with a
point's importance, and that divergence is what identifies them apart.

- **Serve-transition predicts a flat gradient.** The next-point advantage conferred by
  winning is the serve itself, and the value of the serve (servers win ≈60–65% of
  points) does not depend on how important the won point was. Winning a trivial point
  and winning a pivotal point, if both transfer serve, should confer the same
  structural boost. The effect should be approximately constant across leverage.

- **Discouragement predicts a rising gradient.** This is Gauriot and Page's own
  finding: the effect scales with point importance, from ≈2.5 pp at low-stakes states
  to ≈10.6 pp at symmetric high-stakes states, because the continuation-value asymmetry
  that drives rational effort-reduction is itself largest where stakes are highest.

So the test is a single, pre-specified, falsifiable question: **does the estimated
effect of winning scale with the leverage of the point won, or stay flat?** A flat
gradient is evidence for serve-transition; a rising gradient replicating the
Gauriot–Page slope is evidence that our effect is partly discouragement.

#### 7a.3 The test

The quantity of interest is the heterogeneity of the *already-estimated* treatment
effect across the leverage of the treated point — a conditional-effect gradient, not a
new estimator. Concretely, leverage enters as a covariate and the treatment effect is
read as a function of it, either by reading the causal-forest CATE against the leverage
axis or by re-estimating the ATE within leverage strata. No new identification
machinery is introduced; the design of §4 is reused with one added covariate.

**Design fork on the treatment definition — resolved.** The headline treatment (won
high-leverage point) is, by construction, concentrated at the upper end of the leverage
range. Two options were considered:

1. Read the existing won-HLP effect across the leverage it does span (narrower range,
   no change to the treatment, directly interpretable against the headline).
2. Broaden the treatment to *winning any point*, estimated across the full leverage
   gradient, matching Gauriot and Page's own design.

**Option 1 is primary; option 2 is a positive control, not a co-primary test.** Option 1
is the direct defence — it tests heterogeneity in the exact estimand the paper's spine
rests on. Making the redefined, broader treatment of option 2 the primary discriminating
test would invite the objection that the test answers "does momentum-in-general scale
with leverage," not "does *our* break-point effect scale with leverage" — a different
question that happens to sit on Gauriot and Page's own terrain. Option 2's proper role is
validating the leverage measure itself: since Gauriot and Page's rising gradient is an
already-established finding on the *winning-any-point* population, replicating it there
confirms the instrument can detect a real gradient when one exists. That licenses trusting
a null from the same instrument on option 1. The asymmetry that follows matters and should
stay visible in the write-up: **a successful option 2 (replicating the known rising
gradient) validates the measure and supports treating option 1's result at face value; a
failed option 2 (flat where a gradient is known to exist) is inconclusive about option 1,
not evidence against it** — it would mean the instrument lacks power or specificity, not
that Gauriot and Page's finding fails to replicate. Report both, primary result from
option 1, option 2 as the validity check that licenses trusting it.

**Leverage-spread pre-check (done, on existing data, no new fitting).** Checked whether
the won-break-point treated population (option 1's treatment) spans enough leverage
internally to make the test well-posed. Three axes, using
`data/processed/{atp,wta}_cleaned_points.csv` restricted to `High_Leverage_BP==1 &
Point_Won==1` (1,497 ATP / 1,001 WTA):

| axis | spread |
|---|---|
| Within-game score state (`Pts`) | Only 4 discrete values by construction (0-40/15-40/30-40/40-AD) — every treated point *is* a break point, so this axis alone is genuinely coarse (non-degenerate: 10-36% each, but 4 levels, not a gradient). |
| Game-score margin in the set (`\|Gm1-Gm2\|`) | Good spread: 0-5, median 1, IQR [0, 2]. Break points occur across both tight and lopsided set states. |
| Set number | Spans all sets with reasonable weight (ATP 0-4, WTA 0-2), not concentrated in one set. |

**Conclusion: option 1 is well-posed, but not on the within-game-closeness proxy alone as
§7a.4 currently describes it.** That axis alone gives only 4 buckets for the treated
population specifically (by construction of what a break point is). Game-margin and
set-number carry real, currently-unused spread. §7a.4's proxy needs broadening to
incorporate match/set context, not just point-level score state, before it's fit for
option 1 — a narrow proxy would make option 1 look under-powered when the actual
constraint is the leverage *definition*, not the data.

The test runs on the **full regular-play point population**, where power is abundant
(tens of thousands of points), rather than on the sparse tiebreak cells. This is
deliberate: the objection lives on the *positive* effect, where the data is thick, so
the test attacks it there rather than in the underpowered tiebreak cells of §5–§6.

#### 7a.4 The leverage measure

**Two-speed by design.** Reconnaissance uses a coarse score-state proxy: importance
buckets built from within-game closeness and lateness (deuce / 30–30 / 40–30 high;
0–0 / 40–0 low), deliberately constructed to approximate the Gauriot–Page gradient so
that agreement with the principled measure later is confirmation and disagreement is
signal. This runs on existing infrastructure and delivers a first look at flat-vs-rising
within a day's work.

The shipped version uses **Morris (1977) point importance** as formalised by Klaassen
and Magnus (2001): the leverage of a point is the difference in the player's probability
of winning the game (or set) conditional on winning versus losing that point — the
win-probability swing. This is the field-standard measure, it is the exact language of
the objection, and a referee will expect it rather than a home-rolled proxy. The proxy
is reconnaissance; Morris importance is what ships. The first look must not be blocked
on building the win-probability model, and the proxy result must not become the final
answer.

#### 7a.5 The within-tiebreak partition (corroborating check only)

A second, weaker check partitions tiebreak points by leverage and asks whether the null
holds across the partition. Its role is corroboration, not primary evidence, for two
reasons. First, it inherits and amplifies the power limitation of §6: splitting 729 ATP
(and 209 WTA) treated tiebreak points into leverage subsets yields cells too thin to
carry weight, WTA especially. Second, its discriminating power is asymmetric — a masked
effect would re-emerge most at *low*-leverage tiebreak points, which is the
better-populated subset, so the low-leverage null is the load-bearing arm and the
high-leverage null is corroborating-but-thin. This check is reported for completeness
and consistency with the primary leverage-scaling test, not as independent proof.

#### 7a.6 What each outcome licenses

This is a genuine fork, and the section is written to collapse to whichever branch the
data delivers — not to a foregone flat gradient.

- **Flat gradient.** Evidence that the break-point effect is the serve transition, which
  is importance-invariant. The spine stands as written, and the result additionally
  distinguishes our mechanism from Gauriot and Page's: theirs scales with leverage, ours
  does not, so we are identifying a different (structural, mechanical) channel rather
  than re-finding strategic momentum.

- **Rising gradient.** Evidence that the effect is partly discouragement. This does *not*
  collapse the falsification logic — the tiebreak null still constrains the psychological
  reading — but it modifies the spine: the honest finding becomes "serve-transition plus
  a strategic-effort component that scales with stakes," a more complicated and less
  clean claim than "it is the serve." If the data lands here, the spine is rewritten to
  say so; it is not reframed until it looks flat.

The sex-asymmetry connection (see §6) is adjudicated by the same test where WTA power
permits. Serve-transition is sex-neutral (serve rotates identically on both tours);
discouragement is plausibly male-specific too, by analogy to a documented, male-specific
set-level winner effect in tennis: Page and Coates (2017) find that among equally-matched
players (restricting to matches where the first set went to a long tiebreak, >20 points,
so the set winner is quasi-random with respect to ranking), the winner of that tiebreak
takes the second set ~60% of the time among men, versus ~51% among women (~53% to the
loser) — effectively absent. They propose an androgen-mediated mechanism (testosterone
elevating confidence and performance in the following set). **This reference supports
two things here and not a third: that a winner effect in tennis is documented, and that
it is male-specific. It does not support, and this document does not claim, that our
point-level mechanical account overturns or subsumes their set-level physiological one
— different level of analysis, different mechanism, no refutation implied.** [citation
verified against project notebook extract, not the primary source; see §7a.7 item 5.]
A flat WTA gradient would support reading the WTA break-point effect as sex-neutral
structural rather than as a fragile artifact; a rising or uninformative WTA gradient
would send the WTA result back to the underpowered-caveat treatment of §6. This is
flagged as a question the test bears on, **not** as a contribution claim — the WTA
cells may be too thin to answer it, and the flattering interpretation is not to be
assumed.

#### 7a.7 Open dependencies and unverified claims

1. **Proxy result pending.** The flat-vs-rising question is unanswered until the
   reconnaissance test is run. No sentence in this section asserts the answer.
2. **Morris measure to replace the proxy before ship.** The proxy is reconnaissance
   only; the win-probability-swing measure is required for the methodology document and
   the paper.
3. **Resolved: treatment-definition fork.** Option 1 (won-HLP-across-its-range) is
   primary; option 2 (won-any-point-across-full-leverage) is a positive control on the
   leverage measure, not a co-primary test — see §7a.3 for the reasoning and the
   asymmetry (successful option 2 validates the measure; a failed option 2 is
   inconclusive about option 1, not damning). Leverage-spread pre-check confirmed
   option 1 is well-posed, conditional on item 3a below.
3a. **New: the leverage proxy (§7a.4) needs broadening before it's fit for option 1.**
   The spread check found the within-game-closeness axis alone gives only 4 discrete
   buckets for the won-break-point population (break points are, by construction, an
   already-narrow slice of within-game states). Game-score margin and set number carry
   real, currently unused spread. §7a.4's proxy description needs to incorporate
   match/set context, not just point-level score state, before the reconnaissance test
   runs on option 1.
4. **UNVERIFIED mechanistic claim:** the assertion that discouragement *persists in
   tiebreaks* (and that the tiebreak null therefore also rebuts discouragement) has
   **not** been checked against Gauriot and Page's actual continuation-value model. It is
   a plausible hypothesis, not an established joint, and must not be load-bearing until
   verified against the paper's formalism. The leverage-scaling test does not depend on
   it, which is a further reason to keep that test primary.
5. **Citation now sourced (§7a.6):** Page, L., & Coates, J. M. (2017). Winner and loser
   effects in human competitions. Evidence from equally matched tennis players.
   *Evolution and Human Behavior*, 38(4), 530–535. Correctly scoped to what it
   supports (a documented, male-specific winner effect) and explicitly not claimed
   to support a refutation of their mechanism — see the bracketed scope note at
   §7a.6. **Provenance note:** the Page & Coates and Gauriot & Page citations in
   this section are verified against project notebook extracts, not the primary
   sources directly. That's an adequate standard for this draft; before journal
   submission, the 60/40, 51/53, >20-point-tiebreak threshold (Page & Coates) and
   the 7.2pp / 2.5pp / 10.6pp figures (Gauriot & Page) should be checked once
   against the actual papers, same as any other load-bearing quote.

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
committed. §7a (strategic-momentum objection) is a full provisional draft,
not yet actionable. The treatment-definition fork is resolved (option 1
primary, option 2 as positive control, asymmetry stated — §7a.3), confirmed
well-posed by a leverage-spread pre-check on existing data. What remains,
listed at §7a.7: the leverage proxy needs broadening beyond within-game score
state before it runs (item 3a — the spread check found that axis alone gives
only 4 buckets for the treated population); the reconnaissance proxy hasn't
been built or run; the Morris/Klaassen-Magnus importance measure isn't built;
and the discouragement-in-tiebreaks claim is unverified against Gauriot &
Page's actual model, with no confirmed path to primary-source access yet.
The Page & Coates (2017) citation in §7a.6 is sourced and correctly scoped
(documented male-specific winner effect; explicitly not a refutation of their
mechanism) — it and the Gauriot & Page figures are verified against project
notebook extracts, with a primary-source spot-check owed before journal
submission, not before this draft. Nothing in §7a should be treated as
analysis-ready until the remaining open items clear.*
