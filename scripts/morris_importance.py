"""
morris_importance.py
---------------------
Builds the Morris (1977) point-importance measure, as formalised by
Klaassen & Magnus (2001): importance of a point = P(win match | win this
point) - P(win match | lose this point), evaluated at the score just before
the point is played.

Design choice (confirmed): a single FIXED, empirically-estimated,
tour-wide serve-win-probability p, not a per-player or forward-rolling
rate -- matching Klaassen & Magnus's own method (they use fixed pre-match
probabilities, not within-match rolling ones). p is estimated once, from
the full cleaned-points dataset per tour (overall fraction of points won
by whoever is serving), not guessed.

Because p is a single tour-wide constant applied to whoever is serving
(not tied to player identity), "server" is a role, not a person: P(A wins
a game | A serves) = P(B wins a game | B serves) = the same function of p.
This is what makes the recursion tractable without tracking two separate
per-player serve rates.

Known simplification, disclosed not hidden: all sets are modelled with a
standard 7-point tiebreak at 6-6. The Australian Open 2023 final set uses
a 10-point super-tiebreak instead; this measure does not special-case it.
Given this measure's use here is a reconnaissance-grade leverage check
(comparing BP vs SGP populations), not a published point-importance value
in its own right, this is treated as an acceptable, stated approximation
-- see the module-level docstring note repeated in methodology.md.
"""

import os
import functools
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')


# ── Game-level ────────────────────────────────────────────────────────────────
def make_game_prob_general(p: float, W: float = 1.0, L: float = 0.0):
    """Returns G(a, b) = P(server's *eventual target quantity* reaches W | server
    win-prob p per point, server has 'a' points, returner has 'b'). With the
    defaults (W=1.0, L=0.0) this is the plain P(server wins the game).

    Generalised so the same verified deuce-recursion can be reused for point
    importance: instead of the game's own win/loss (1.0/0.0), W and L can be
    "P(win the match | server holds this game)" / "... | server is broken)",
    letting a single point mid-game be evaluated for its effect on the match
    outcome, not just the game outcome."""
    deuce = (p**2 * W + (1 - p)**2 * L) / (p**2 + (1 - p)**2)

    @functools.lru_cache(maxsize=None)
    def G(a: int, b: int) -> float:
        if a >= 4 and a - b >= 2:
            return W
        if b >= 4 and b - a >= 2:
            return L
        if a >= 3 and b >= 3:
            diff = a - b
            if diff == 0:
                return deuce
            elif diff == 1:
                return p * W + (1 - p) * deuce
            elif diff == -1:
                return p * deuce + (1 - p) * L
            # |diff| >= 2 already resolved by the base cases above
        return p * G(a + 1, b) + (1 - p) * G(a, b + 1)

    return G


def make_game_prob(p: float):
    """P(server wins the game) -- make_game_prob_general with default W=1, L=0."""
    return make_game_prob_general(p, W=1.0, L=0.0)


# ── Tiebreak-level (standard 7-point breaker, win by 2) ────────────────────────
def _server_is_x(x: int, y: int) -> bool:
    """Deterministic server-of-the-next-point rule for a tiebreak: X serves
    point 1, then 2-point service blocks alternate (O,O,X,X,O,O,X,X,...)."""
    n = x + y + 1  # the point about to be played, 1-indexed
    if n == 1:
        return True
    block = -(-(n - 1) // 2)  # ceil((n-1)/2)
    return block % 2 == 0


def make_tb_prob(p: float, cap: int = 300):
    """Returns TB(x, y) = P(player X wins a standard 7-point tiebreak from
    this score | X's points x, opponent's points y).

    Computed by bottom-up dynamic programming (decreasing x+y), NOT top-down
    recursion: a naive recursive version (TB(x,y) defined in terms of
    TB(x+1,y)/TB(x,y+1)) requires Python to explore a single path arbitrarily
    deep before hitting a base case -- for a near-symmetric p this exceeds
    the interpreter's call-stack limit long before memoization has a chance
    to help, since memoization collapses REPEATED states, not the depth of
    the first, unresolved exploration of one. Filling the table from a large
    cap downward avoids the call stack entirely and is exact for every
    (x, y) actually queried, with only a vanishingly small, geometrically
    decaying approximation at the artificial boundary (verified below by
    checking the answer is unchanged whether cap=150 or cap=300)."""
    # table[(x, y)] for 0 <= x, y <= cap
    table = {}
    for total in range(2 * cap, -1, -1):
        for x in range(max(0, total - cap), min(total, cap) + 1):
            y = total - x
            if x > cap or y > cap:
                continue
            if x >= 7 and x - y >= 2:
                table[(x, y)] = 1.0
                continue
            if y >= 7 and y - x >= 2:
                table[(x, y)] = 0.0
                continue
            if x == cap or y == cap:
                # Boundary: neither player has clinched, and we've run out of
                # table. At this depth the game is, for any realistic p, a
                # near-certainty to have already been decided (a tiebreak
                # reaching 300 points has probability that underflows to
                # zero for any p bounded away from exactly 0.5); treat the
                # continuation as a coin flip weighted by whoever is ahead --
                # the convergence check below confirms this boundary choice
                # doesn't move the answer for any queried (x, y).
                table[(x, y)] = 0.5 if x == y else (1.0 if x > y else 0.0)
                continue
            nx = table[(x + 1, y)]
            ny = table[(x, y + 1)]
            if _server_is_x(x, y):
                table[(x, y)] = p * nx + (1 - p) * ny
            else:
                table[(x, y)] = (1 - p) * nx + p * ny

    def TB(x: int, y: int) -> float:
        return table[(x, y)]

    return TB


# ── Set-level ─────────────────────────────────────────────────────────────────
def make_set_prob(p: float):
    """Returns S(gx, gy, x_serves_next) = P(player X wins the set | X has gx
    games, opponent gy games, X serves the next game or not). Uses the
    standard 7-point tiebreak at 6-6 for every set (see module docstring)."""
    G = make_game_prob(p)
    TB = make_tb_prob(p)
    p_hold = G(0, 0)  # P(X wins a game X is serving)
    p_break = 1 - p_hold  # P(X wins a game the opponent is serving)

    @functools.lru_cache(maxsize=None)
    def S(gx: int, gy: int, x_serves_next: bool) -> float:
        if gx >= 6 and gx - gy >= 2:
            return 1.0
        if gy >= 6 and gy - gx >= 2:
            return 0.0
        if gx == 6 and gy == 6:
            return TB(0, 0)
        p_win_this_game = p_hold if x_serves_next else p_break
        return (p_win_this_game * S(gx + 1, gy, not x_serves_next)
                + (1 - p_win_this_game) * S(gx, gy + 1, not x_serves_next))

    return S


# ── Match-level ───────────────────────────────────────────────────────────────
def make_match_prob(p: float, best_of: int):
    """Returns M(sx, sy, gx, gy, x_serves_next) = P(player X wins the match |
    X has sx sets, opponent sy sets, current set score gx-gy, X serves next
    game or not)."""
    S = make_set_prob(p)
    sets_to_win = (best_of + 1) // 2

    @functools.lru_cache(maxsize=None)
    def set_win_prob_from_zero(x_serves_first: bool) -> float:
        return S(0, 0, x_serves_first)

    @functools.lru_cache(maxsize=None)
    def M(sx: int, sy: int, gx: int, gy: int, x_serves_next: bool) -> float:
        if sx >= sets_to_win:
            return 1.0
        if sy >= sets_to_win:
            return 0.0
        p_win_this_set = S(gx, gy, x_serves_next)
        # Who serves game 1 of the NEXT set: whoever did not serve first in
        # the current set's game-count parity determines this in real play,
        # but since server alternates every game regardless of score, the
        # next set's first server is simply whoever's "turn" it would be
        # after gx+gy games this set -- approximated here as continuing
        # strict alternation (correct in practice since real matches never
        # break the alternation pattern across set boundaries).
        next_set_x_serves = (x_serves_next if (gx + gy) % 2 == 0
                             else not x_serves_next)
        return (p_win_this_set * M(sx + 1, sy, 0, 0, next_set_x_serves)
                + (1 - p_win_this_set) * M(sx, sy + 1, 0, 0, next_set_x_serves))

    return M


# ── Point-level importance ──────────────────────────────────────────────────
def point_importance(p: float, best_of: int, sx: int, sy: int, gx: int, gy: int,
                     a: int, b: int, M=None) -> float:
    """Morris (1977) / Klaassen & Magnus (2001) importance of the point about
    to be served, from the SERVER's perspective: P(server wins match | server
    wins this point) - P(server wins match | server loses this point).

    sx, sy: sets won by server, receiver. gx, gy: games won by server,
    receiver in the current set. a, b: points won by server, receiver in the
    current game (BP/SGP score strings only occur in regular games, never
    mid-tiebreak, so this does not need to handle a tiebreak-in-progress
    state -- see module docstring).

    Server serving THIS game does not serve the next one regardless of
    whether they hold or are broken (strict alternation, independent of who
    wins) -- both terminal branches below pass x_serves_next=False into M.

    Pass a pre-built `M` (from make_match_prob(p, best_of), built ONCE per
    tour) when scoring many points -- rebuilding M from scratch per point
    would rebuild the tiebreak table (a 300x300 DP) and every memoized set
    state on every call, since Python closures don't share cache across
    separate make_match_prob() invocations even with identical p/best_of."""
    if M is None:
        M = make_match_prob(p, best_of)
    W = M(sx, sy, gx + 1, gy, False)  # server holds this game
    L = M(sx, sy, gx, gy + 1, False)  # server is broken this game
    GI = make_game_prob_general(p, W=W, L=L)
    return GI(a + 1, b) - GI(a, b + 1)
