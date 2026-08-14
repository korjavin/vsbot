//! AlphaZero self-play generation: MCTS-vs-MCTS games written as trainer rows.
//!
//! This crate is the Rust replacement for Java's
//! `nnue-trainer/.../mcts/SelfPlayMcts.java`. It replaces the **emitter only** —
//! `python/mcts/train_selfplay.py` stays byte-identical and consumes what this
//! writes without knowing which language produced it. `trainer/README.md`
//! states that constraint; everything here follows from it.
//!
//! ```no_run
//! use virus_mcts::PolicyValueNet;
//! use virus_selfplay::{generate, Options};
//!
//! let net = PolicyValueNet::load("artifacts/mcts_champion.json")?;
//! let mut out = std::io::stdout().lock();
//! let stats = generate(&Options::default(), Some(&net), &mut out)?;
//! eprintln!("{} rows over {} games", stats.rows, stats.games);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # The row contract, and why each half of it is easy to get wrong
//!
//! One JSON object per line, keys in this order:
//!
//! ```text
//! {"g":"sp11011-3", "sym":[144 ints], "ml":1..3, "nuo":0|1, "nux":0|1,
//!  "mover":1|2, "pi":[flat ids], "pv":[root visits], "z":-1|0|1}
//! ```
//!
//! `trainer/validate_rows.py` is the executable statement of this contract and
//! is the acceptance gate for this crate. The four traps it exists to catch,
//! and how this crate avoids each:
//!
//! * **`z` is ABSOLUTE, not mover-relative.** `+1` means player 1 won, on every
//!   row of the game including the ones where player 2 is to move. The trainer
//!   flips it per row via `"mover"`. Pre-flipping here would train the value
//!   head backwards for one seat and *nothing downstream would complain*. This
//!   crate writes [`virus_mcts::terminal_value_abs`] straight through, once per
//!   game, into every row of that game — there is no per-row `z` computation to
//!   get wrong.
//! * **`pi` is every legal action, not the searched subset.** It is the mask as
//!   well as the target's index set, so it comes from
//!   [`virus_mcts::MctsSearcher::root_actions`] — the root's full enumeration —
//!   never from a filtered or top-k view.
//! * **`pv` is raw visit counts, not probabilities.** The trainer normalises.
//!   [`virus_mcts::MctsSearcher::root_visits`] is already the raw `u32` counts.
//! * **Pair ids are ordered.** `144 + min*144 + max`, done by
//!   [`virus_mcts::action_id`], which is shared with the searcher's own policy
//!   lookup — so an emitted id and the id the net was trained against cannot
//!   drift apart.
//!
//! Rows are emitted only for **multi-choice** positions — those with more than
//! one entry in `root_actions()`. A forced position carries no policy signal
//! and only dilutes the target.
//!
//! # Determinism
//!
//! The acceptance criterion is "deterministic per `(seed, shard)` regardless of
//! shard count", which is a stronger statement than "reproducible". It requires
//! that a game's entire random stream derive from `(seed, game_index)` and
//! nothing else — not from the shard index, not from the shard count, not from
//! how many games ran before it in this process. So:
//!
//! * game `g` is played by a shard iff `g % shard_count == shard_idx`, and the
//!   loop still walks `0..games` in every shard (the `V3DeepLabelEmitter`
//!   pattern Java uses), so `g` means the same game everywhere;
//! * the game seed is [`mix64`]`(seed ^ GOLDEN_GAMMA * (g + 1))`;
//! * each ply's searcher is seeded [`mix64`]`(game_seed ^ (ply + 1))`, so even
//!   the per-ply stream is a pure function of `(seed, g, ply)`.
//!
//! Splitting one shard into four therefore repartitions the *same* games rather
//! than generating different ones, and `cat`ting the shards back together gives
//! the same set of rows a single shard would have written.
//!
//! ## Deliberate deviation from Java: the stream, not the derivation
//!
//! Java draws its visit-sampling coin from a separate `java.util.Random` seeded
//! `mix64(gameSeed ^ 0x5DEECE66D)`. This crate lets
//! [`virus_mcts::MctsSearcher::chosen_action`] draw from the searcher's own
//! SplitMix64 stream instead. Byte-identical replay of a Java self-play game is
//! unreachable regardless — the two searchers differ, so the games diverge at
//! the first sampled ply — and reimplementing `java.util.Random` with no JVM on
//! the build hosts to check it against would be a liability with no payoff.
//! `virus-arena`'s `rng` module records the same decision for the same reason.
//! What is load-bearing is preserved: one stream per `(game, ply)`, seeded from
//! the game seed, with draws taken in a fixed order.
//!
//! # The turn cap is not a draw
//!
//! `virus-arena` and the `batchgauntlet` example score a turn-capped game as a
//! draw, which is right for a *gauntlet*: a game neither side could finish is
//! not evidence either way. It is wrong for *training data*. A capped game is
//! still a position with a territory majority, and labelling it `0` teaches the
//! value head that a comfortably-won-but-slow position is balanced. Java's
//! generator scores the final state with `outcomeWinner()` whether it ended
//! naturally or hit the cap, and so does this crate — see [`play_game`].
//!
//! # Resignation (off by default)
//!
//! [`ResignConfig`] is the AZ throughput lever: a game whose loser is already
//! decided is a played-out tail that costs a search per ply and teaches the net
//! nothing it does not already know. When [`GameConfig::resign`] is `Some`, a
//! game stops early once the side to move has been hopeless for long enough.
//!
//! Three things about it are load-bearing, and each is a way to poison a
//! generation rather than merely shorten it:
//!
//! * **`z` stays honest and stays absolute.** A resigned game is scored
//!   "the side that did *not* resign won" — `-1` when player 1 resigns, `+1`
//!   when player 2 does. It is *not* scored from the position on the board
//!   (which is unfinished, and whose territory verdict may still favour the
//!   resigning side) and it is *not* the search's value estimate rounded off.
//!   The turn-cap and natural-end paths are untouched.
//! * **Resignation only truncates; it never perturbs.** The resign test reads
//!   [`MctsSearcher::root_value_abs`] after the search that ply would have run
//!   anyway and draws no randomness. So game `g` with resign on plays the *same
//!   moves* as game `g` with resign off, up to the ply it stops at, and its rows
//!   are that game's rows truncated — same bytes, except for `z` when the
//!   resignation was wrong. `resign_truncates_rather_than_perturbing` pins this.
//! * **The control arm is what makes the feature admissible.** A resign rule
//!   that fires on positions the resigning side would have *won* silently
//!   mislabels those games. [`ResignConfig::control_frac`] of games (10 % by
//!   default) ignore the rule and play out, recording where resignation *would*
//!   have fired; comparing that counterfactual against the real result is the
//!   false-resign rate, and the AZ bar is <5 %. Control membership is drawn from
//!   the game seed ([`is_control_game`]), so it is a pure function of
//!   `(seed, game_index)` like everything else here — same shards, same arms.
//!
//! The rule itself: at each ply, the side to move is "hopeless" when the root's
//! absolute-frame value is beyond [`ResignConfig::threshold`] *against* it
//! (`v < -threshold` for player 1, `v > threshold` for player 2). A per-side
//! counter of consecutive hopeless plies **by that side** — the opponent's plies
//! neither increment nor reset it — reaches [`ResignConfig::plies`] and the game
//! ends. Counting a side's own plies is why `plies` is a small number in a game
//! where the mover does not alternate (invariant 1): 8 own plies is under three
//! of that side's turns, but spans roughly sixteen plies of play.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

use std::io::{self, Write};
use std::time::Instant;

use serde::Serialize;
use virus_core::{Player, State, ACTIONS_PER_TURN, MAX_PLAYERS};
use virus_mcts::{
    action_id, net::Encoded, terminal_value_abs, Config, MctsSearcher, PolicyValueNet,
};

/// The golden-ratio odd constant SplitMix64 strides by, and the multiplier in
/// the per-game seed derivation. Matches Java's `SelfPlayMcts`.
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Default simulations per action.
///
/// Java's production generation used `SIMS=256`; 192 is this project's default
/// because it is the largest count that keeps a 12x12 self-play game inside a
/// couple of minutes on one core of the shared devbox, and the sims/action knob
/// is the one a generation script always overrides anyway.
pub const DEFAULT_SIMS: u32 = 192;

/// Default turn cap, matching Java's `MAX_PLIES = 100 * ACTIONS_PER_TURN`.
pub const DEFAULT_MAX_TURNS: u32 = 100;

/// Default resign threshold: the AZ paper's 0.95, on this project's `[-1, 1]`
/// absolute-frame value.
pub const DEFAULT_RESIGN_THRESHOLD: f64 = 0.95;

/// Default consecutive hopeless plies **by the resigning side** before a game
/// is resigned.
pub const DEFAULT_RESIGN_PLIES: u32 = 8;

/// Default share of games that ignore the resign rule and play out, so the
/// false-resign rate is measurable. AZ's figure, and the denominator of the
/// <5 % promotion bar.
pub const DEFAULT_RESIGN_CONTROL_FRAC: f64 = 0.10;

/// Salt for the control-arm coin.
///
/// The coin has to be independent of every other draw a game makes, and it must
/// not *consume* from any of them: if choosing the arm advanced the searcher's
/// stream, a control game and its resign-enabled twin would play different
/// moves and the counterfactual the arm exists to measure would be comparing
/// two unrelated games. So it is one extra [`mix64`] of the game seed under a
/// salt of its own — a pure function of `(seed, game_index)` that touches
/// nothing.
const CONTROL_SALT: u64 = 0x5265_7369_676E_4B21; // "ResignK!"

/// When to give a hopeless game up, and how often not to.
///
/// See the module docs for the rule and for why the control arm is not
/// optional. Absent from [`GameConfig`] (`resign: None`) the whole mechanism is
/// inert: no counters, no arms, no behaviour change of any kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResignConfig {
    /// How far beyond hopeless, in the absolute-frame value, a side has to be.
    /// `0.95` means "worse than -0.95 for player 1, better than +0.95 for
    /// player 2". Compared strictly, so `1.0` is effectively "never".
    pub threshold: f64,
    /// Consecutive plies **by the side to move** at or beyond `threshold`
    /// against it before the game ends. The opponent's plies in between neither
    /// increment nor reset the count.
    pub plies: u32,
    /// Share of games in `[0, 1]` that ignore the rule and play out. `0.0`
    /// disables the control arm, which also disables the only measurement that
    /// can justify turning resignation on — do it only when the rate has
    /// already been measured and cleared.
    pub control_frac: f64,
}

impl Default for ResignConfig {
    fn default() -> ResignConfig {
        ResignConfig {
            threshold: DEFAULT_RESIGN_THRESHOLD,
            plies: DEFAULT_RESIGN_PLIES,
            control_frac: DEFAULT_RESIGN_CONTROL_FRAC,
        }
    }
}

/// Whether game `game` of seed `seed` is in the no-resign control arm.
///
/// A pure function of `(seed, game)` — not of the shard index, not of the shard
/// count, not of the resign settings. Re-sharding a run therefore repartitions
/// the same arms rather than redrawing them, which is what lets two runs at
/// different shard counts be compared row for row.
#[must_use]
pub fn is_control_game(seed: u64, game: u64, control_frac: f64) -> bool {
    control_draw(derive_game_seed(seed, game)) < control_frac
}

/// The control coin for a game seed, uniform in `[0, 1)`.
///
/// Built from the top 53 bits so it is exactly representable as an `f64`, the
/// standard construction — the low bits of a SplitMix64 finalizer are the ones
/// worth discarding, and using them here would correlate the arm with the least
/// avalanched part of the seed.
#[must_use]
fn control_draw(game_seed: u64) -> f64 {
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    (mix64(game_seed ^ CONTROL_SALT) >> 11) as f64 * SCALE
}

/// SplitMix64's finalizer: a bijection on `u64` with full avalanche.
///
/// The mixing half of SplitMix64 with no state increment, identical constant
/// for constant to Java's `SelfPlayMcts.mix64` and to `virus_arena::rng::mix64`.
/// It is reimplemented here rather than imported because this crate depends on
/// the engine crates only — `virus-arena` is a consumer of engines, and taking
/// a dependency on the gauntlet harness to reach three lines of arithmetic
/// would invert the dependency direction CLAUDE.md fixes.
#[must_use]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed for game `g` of a run: a pure function of `(seed, g)`.
///
/// Deliberately *not* a function of the shard index or count — that is the
/// whole of the "deterministic per `(seed, shard)` regardless of shard count"
/// requirement. Unlike `virus_arena::rng::derive_game_seed` there is no `g / 2`
/// pair index here: self-play games are not colour-paired, both seats being the
/// same engine, so every game gets its own opening.
#[must_use]
pub fn derive_game_seed(seed: u64, game: u64) -> u64 {
    mix64(seed ^ GOLDEN_GAMMA.wrapping_mul(game.wrapping_add(1)))
}

/// One training row, serialised as one JSONL line.
///
/// Field order is the serialised key order and matches Java's `row()`. See the
/// module docs for what each field means and which of them are traps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Row {
    /// Game id, `sp<seed>-<game index>`. Every row of a game shares it *and*
    /// shares `z`.
    pub g: String,
    /// The 144 `PatternContract` cell symbols, row-major, **mover-relative**.
    pub sym: Vec<u8>,
    /// Actions remaining in the current turn, `1..=3`.
    pub ml: u8,
    /// `1` when the mover has already spent its neutral placement.
    pub nuo: u8,
    /// `1` when the opponent has already spent its neutral placement.
    pub nux: u8,
    /// Whose turn it is; the trainer's `z` flip point.
    pub mover: Player,
    /// Flat ids of **every** legal root action — the policy mask.
    pub pi: Vec<u32>,
    /// Raw root visit counts, parallel to `pi` — the policy target.
    pub pv: Vec<u32>,
    /// Game outcome in the **absolute** frame: `+1` player 1 won, `-1` player
    /// 2 won, `0` drawn. Filled in when the game ends, identical on every row.
    pub z: i8,
}

/// Per-game search settings.
#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    /// Simulations per action.
    pub sims: u32,
    /// PUCT exploration constant.
    pub cpuct: f64,
    /// Divisor in the hand-tuned leaf's `tanh` squash.
    pub value_scale: f64,
    /// Leaves per batched net evaluation. Changes the tree (and so the games),
    /// but not their determinism.
    pub batch_size: u16,
    /// Turn cap. A capped game is scored by territory, not called a draw.
    pub max_turns: u32,
    /// Resignation, or `None` for "play every game out" — the default, and what
    /// every generation to date ran with. See the module docs.
    pub resign: Option<ResignConfig>,
}

impl Default for GameConfig {
    fn default() -> GameConfig {
        let template = Config::self_play(0, 0);
        GameConfig {
            sims: DEFAULT_SIMS,
            cpuct: template.cpuct,
            value_scale: template.value_scale,
            batch_size: template.batch_size,
            max_turns: DEFAULT_MAX_TURNS,
            resign: None,
        }
    }
}

/// What one self-play game produced.
#[derive(Clone, Debug)]
pub struct GameReport {
    /// The game's rows, every one already carrying the final `z`.
    pub rows: Vec<Row>,
    /// Actions played.
    pub plies: u32,
    /// Searches run. One more than [`GameReport::plies`] whenever the game
    /// stopped *after* a search rather than before one (resignation, or a stuck
    /// root), and equal to it otherwise. This — not `plies` — is what a game
    /// costs, so it is the honest denominator for a throughput claim.
    pub searches: u32,
    /// Absolute-frame outcome.
    pub z: i8,
    /// Whether the game stopped at the turn cap rather than finishing.
    pub capped: bool,
    /// The side that resigned, when the game ended by resignation. `None` for
    /// every game played out — including every control-arm game, which is why
    /// its `z` is the board's verdict and not a resignation verdict.
    pub resigned: Option<Player>,
    /// Whether this game was in the no-resign control arm. Always `false` when
    /// [`GameConfig::resign`] is `None`: a run with no resign rule has no arms.
    pub control: bool,
    /// For a control game, the side that *would* have resigned and the ply it
    /// would have happened on — the counterfactual the false-resign rate is
    /// computed from. `None` on control games the rule never fired for, and on
    /// every non-control game (where it is not a counterfactual but the actual
    /// [`GameReport::resigned`]).
    pub would_resign: Option<(Player, u32)>,
}

/// Plays one self-play game from the production start position.
///
/// The loop is Java's `playGame`, action for action:
///
/// 1. build a fresh searcher for the position, seeded `mix64(game_seed ^ (ply +
///    1))`, with Dirichlet root noise on and temperature-1 visit sampling for
///    the first [`virus_mcts::TEMPERATURE_PLIES`] plies — that is exactly
///    [`Config::self_play`], which is why this crate needs no self-play mode of
///    its own;
/// 2. run `sims` simulations;
/// 3. record a row if the root offered more than one action;
/// 4. play [`MctsSearcher::chosen_action`] (a visit-proportional draw inside
///    the temperature window, the argmax after it).
///
/// A fresh searcher per ply rather than [`MctsSearcher::rebase`] is not an
/// oversight: rebasing would carry the *previous* ply's Dirichlet noise into
/// this root's priors, and root noise is meant to be redrawn at every root.
/// Java rebuilds for the same reason.
///
/// When the loop ends — naturally, at the turn cap, at a stuck root, or by
/// resignation — the outcome is settled once and that single value is stamped
/// onto every pending row. One flip point, in the trainer; see the module docs.
///
/// Resignation is folded into step 2/3 of that loop: the value it tests is the
/// root value of the search the ply ran anyway, so an enabled resign rule adds
/// no work and no randomness — it only decides, after a search, whether there
/// will be another one. The row for the resigning ply is kept: it is a real
/// searched root with a real visit distribution, and dropping it would make the
/// rule change the *content* of a game rather than only its length.
pub fn play_game(
    game_id: &str,
    game_seed: u64,
    net: Option<&PolicyValueNet>,
    config: &GameConfig,
) -> GameReport {
    let mut state = State::new(12, 12, 2).expect("12x12 two-player start position");
    let ply_ceiling = config.max_turns * u32::from(ACTIONS_PER_TURN);
    let mut pending: Vec<Row> = Vec::new();
    let mut ply = 0u32;
    let mut searches = 0u32;
    let mut capped = true;

    let control = config
        .resign
        .is_some_and(|rules| control_draw(game_seed) < rules.control_frac);
    let mut tracker = ResignTracker::default();
    let mut would_resign: Option<(Player, u32)> = None;
    let mut resigned: Option<Player> = None;

    while ply < ply_ceiling {
        if state.game_over() {
            capped = false;
            break;
        }
        let search_config = Config {
            cpuct: config.cpuct,
            value_scale: config.value_scale,
            batch_size: config.batch_size,
            ..Config::self_play(mix64(game_seed ^ u64::from(ply + 1)), ply)
        };
        let mut searcher = MctsSearcher::new(state.clone(), search_config, net);
        searcher.run_sims(config.sims);
        searches += 1;

        // A terminal or stuck root: `game_over()` above did not catch it, so
        // score what we have rather than emitting a row with an empty mask.
        if searcher.root_actions().is_empty() {
            capped = false;
            break;
        }

        if let Some(rules) = config.resign {
            let mover = state.current_player();
            let fired = tracker.observe(mover, searcher.root_value_abs(), &rules);
            // `would_resign` records the *first* firing only: a control game
            // that keeps playing would otherwise re-fire every ply and the
            // counterfactual would no longer be "where the resign arm stopped".
            if fired && would_resign.is_none() {
                would_resign = Some((mover, ply));
                if !control {
                    resigned = Some(mover);
                }
            }
        }

        if searcher.root_actions().len() > 1 {
            pending.push(row(game_id, &state, &searcher));
        }
        if resigned.is_some() {
            capped = false;
            break;
        }
        let Some(action) = searcher.chosen_action() else {
            capped = false;
            break;
        };
        state = state.apply_generated(action);
        ply += 1;
    }

    // A game whose *last allowed* action ended it exits on the loop condition
    // rather than through the `game_over` break, so the flag has to be settled
    // here too. Only the statistic is at stake — `z` is read off the final
    // state either way — but the capped count is how a generation reports
    // whether its turn cap is biting, and a cap that silently absorbs finished
    // games would hide that.
    if state.game_over() {
        capped = false;
    }

    // The one place the board is *not* the authority on the result. A resigned
    // game is unfinished, so `terminal_value_abs` would score its territory —
    // frequently in the resigning side's favour, since resignation fires on a
    // search verdict that runs ahead of the board. Taking the board's answer
    // here would label exactly the games the rule is most confident about as
    // wins for the side that gave up.
    let z = match resigned {
        Some(loser) => {
            if loser == 1 {
                -1
            } else {
                1
            }
        }
        None => sign(terminal_value_abs(&state)),
    };
    for row in &mut pending {
        row.z = z;
    }
    GameReport {
        rows: pending,
        plies: ply,
        searches,
        z,
        capped,
        resigned,
        control,
        would_resign: if resigned.is_some() {
            None
        } else {
            would_resign
        },
    }
}

/// Whether `value`, in the absolute frame, is beyond `threshold` **against**
/// `mover`.
///
/// The sign application invariant 1 warns about, in its smallest form: `+1` is
/// good for player 1 on every row and at every ply regardless of whose turn it
/// is, so "losing badly" is a different comparison for each seat and there is
/// no negamax flip to lean on.
fn is_hopeless(mover: Player, value: f64, threshold: f64) -> bool {
    if mover == 1 {
        value < -threshold
    } else {
        value > threshold
    }
}

/// Consecutive hopeless plies, counted **per seat**.
///
/// One shared counter would be wrong in a way that looks like a working
/// feature: at a ply where player 1 is hopeless the value is far negative, and
/// at the next ply — which invariant 1 says may or may not be player 2's — that
/// same far-negative value means player 2 is comfortable. A shared counter
/// would reset on every mover change and a game would essentially never resign,
/// while every unit test that only ever looks at one seat still passes.
#[derive(Clone, Debug, Default)]
struct ResignTracker {
    /// Indexed by seat number, which is 1-based (`0` means "nobody"), so the
    /// zeroth slot is deliberately dead rather than an off-by-one waiting to
    /// happen. Two seats in self-play, sized for the four the rules allow.
    streaks: [u32; MAX_PLAYERS + 1],
}

impl ResignTracker {
    /// Records one searched ply and reports whether `mover` has just earned a
    /// resignation. Plies by the *other* seat leave this seat's count alone —
    /// they are neither evidence for nor against its position.
    fn observe(&mut self, mover: Player, value: f64, rules: &ResignConfig) -> bool {
        let streak = &mut self.streaks[usize::from(mover)];
        if is_hopeless(mover, value, rules.threshold) {
            *streak += 1;
        } else {
            *streak = 0;
        }
        *streak >= rules.plies
    }
}

/// One row from a searched root. `z` is a placeholder until the game ends.
fn row(game_id: &str, state: &State, searcher: &MctsSearcher) -> Row {
    // `Encoded::from_state` is the searcher's own input encoding, so `sym`,
    // `ml`, `nuo` and `nux` in a row are literally the tensor the net saw. A
    // second, row-only encoder here would be a place for the two to drift.
    let encoded = Encoded::from_state(state);
    Row {
        g: game_id.to_owned(),
        sym: encoded.sym.to_vec(),
        ml: encoded.moves_left,
        nuo: u8::from(encoded.nu_own),
        nux: u8::from(encoded.nu_opp),
        mover: state.current_player(),
        pi: searcher
            .root_actions()
            .iter()
            .map(|action| action_id(*action) as u32)
            .collect(),
        pv: searcher.root_visits().to_vec(),
        z: 0,
    }
}

/// `{-1, 0, 1}` from the terminal value, which only ever holds those three.
///
/// Java writes `(int) terminalValueAbs(state)`; a comparison says the same
/// thing without a float-to-int cast whose rounding would matter if the
/// labelling rule ever grew a fractional value.
fn sign(value: f64) -> i8 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

/// A generation run: which games to play, and how.
#[derive(Clone, Debug)]
pub struct Options {
    /// Global game count. A shard plays the subset it owns, but the indices are
    /// always taken from `0..games`.
    pub games: u64,
    /// Base seed. The game id embeds it, so rows from two seeds never collide.
    pub seed: u64,
    /// This shard's index, `0..shard_count`.
    pub shard_idx: u64,
    /// How many shards the run is split across. `1` means "not sharded".
    pub shard_count: u64,
    /// Per-game search settings.
    pub game: GameConfig,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            games: 8,
            seed: 11,
            shard_idx: 0,
            shard_count: 1,
            game: GameConfig::default(),
        }
    }
}

/// What a generation run produced.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Games this shard played.
    pub games: u64,
    /// Rows written.
    pub rows: u64,
    /// Games that hit the turn cap.
    pub capped: u64,
    /// Outcome tally, indexed as `[player 2 wins, draws, player 1 wins]`.
    pub outcomes: [u64; 3],
    /// Searches run across every game — the shard's cost, in the only unit that
    /// matters at 192 sims a search.
    pub searches: u64,
    /// Games that ended by resignation. Necessarily all from the resign arm.
    pub resigned: u64,
    /// Games in the no-resign control arm.
    pub control: u64,
    /// Control games the rule *would* have resigned.
    pub control_would_resign: u64,
    /// Control games the rule would have resigned where the side that would
    /// have given up did **not** go on to lose. The numerator of the
    /// false-resign rate; its denominator is [`Stats::control_would_resign`].
    ///
    /// A drawn game counts here. "Would not have lost" is the honest reading of
    /// the AZ bar: a resignation that throws away a draw is still a mislabelled
    /// game, and on a board where the turn cap resolves by territory, draws are
    /// rare enough that being strict about them costs nothing.
    pub control_false_resign: u64,
    /// Searches the control arm actually ran, over control games only.
    pub control_searches: u64,
    /// Searches the resign arm would have skipped on those same games — the
    /// only unbiased throughput estimate available, since a resigned game's
    /// full length is by construction unobservable.
    pub control_searches_saved: u64,
    /// Plies the resign arm would have skipped on control games. One more per
    /// resigned game than [`Stats::control_searches_saved`]: the resigning ply
    /// is not played, but its search still happened.
    pub control_plies_saved: u64,
    /// Wall clock.
    pub seconds: f64,
}

impl Stats {
    /// Share of control games the rule would have resigned wrongly, in `[0, 1]`,
    /// or `None` when it never fired and there is nothing to divide.
    ///
    /// This is the number the promotion bar is written against: below `0.05`,
    /// resignation may be turned on for a real generation.
    #[must_use]
    pub fn false_resign_rate(&self) -> Option<f64> {
        (self.control_would_resign > 0)
            .then(|| self.control_false_resign as f64 / self.control_would_resign as f64)
    }

    /// Projected share of a generation's searches that resignation removes, in
    /// `[0, 1]`, or `None` without a control arm to measure it against.
    ///
    /// Measured entirely inside the control arm — saved searches over searches
    /// those games actually ran — because that is the only place both the
    /// resign-arm length and the full length of the *same* game are observable.
    /// Reading it off the resign arm instead (comparing mean game lengths
    /// across arms) would confound the saving with the fact that the games
    /// resignation fires on are the long, decided ones.
    #[must_use]
    pub fn projected_search_saving(&self) -> Option<f64> {
        (self.control_searches > 0)
            .then(|| self.control_searches_saved as f64 / self.control_searches as f64)
    }
}

/// Plays this shard's games and writes their rows to `out` as JSONL.
///
/// Rows are written game by game and flushed after each one, so a run killed
/// part way still leaves a file of whole games — which the validator accepts
/// and the trainer can use. Buffering the whole run would trade that for
/// nothing: the writer is already the cheapest thing in the loop next to a
/// 192-simulation search.
///
/// # Errors
/// Propagates any write or flush error from `out`.
pub fn generate<W: Write>(
    options: &Options,
    net: Option<&PolicyValueNet>,
    out: &mut W,
) -> io::Result<Stats> {
    assert!(options.shard_count > 0, "shard_count must be at least 1");
    assert!(
        options.shard_idx < options.shard_count,
        "shard {} does not exist in a {}-shard run",
        options.shard_idx,
        options.shard_count
    );

    let started = Instant::now();
    let mut stats = Stats::default();
    for g in 0..options.games {
        if g % options.shard_count != options.shard_idx {
            continue;
        }
        let report = play_game(
            &format!("sp{}-{}", options.seed, g),
            derive_game_seed(options.seed, g),
            net,
            &options.game,
        );
        for row in &report.rows {
            serde_json::to_writer(&mut *out, row)?;
            out.write_all(b"\n")?;
        }
        out.flush()?;
        stats.games += 1;
        stats.rows += report.rows.len() as u64;
        stats.capped += u64::from(report.capped);
        stats.outcomes[(report.z + 1) as usize] += 1;
        stats.searches += u64::from(report.searches);
        stats.resigned += u64::from(report.resigned.is_some());
        if report.control {
            stats.control += 1;
            stats.control_searches += u64::from(report.searches);
            if let Some((side, at)) = report.would_resign {
                stats.control_would_resign += 1;
                // The resign arm would have searched plies `0..=at` and played
                // `0..at`, so it stops `searches - (at + 1)` searches and
                // `plies - at` actions short of where this game got to.
                stats.control_searches_saved += u64::from(report.searches - (at + 1));
                stats.control_plies_saved += u64::from(report.plies - at);
                let lost = if side == 1 { -1 } else { 1 };
                stats.control_false_resign += u64::from(report.z != lost);
            }
        }
    }
    stats.seconds = started.elapsed().as_secs_f64();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned SplitMix64 finalizer outputs. Guards the two magic constants
    /// against a transposed digit, which would otherwise show up only as
    /// "reproducible, but not the same games Java would have played".
    #[test]
    fn mix64_matches_the_reference_finalizer() {
        assert_eq!(mix64(0), 0);
        assert_eq!(mix64(1), 6_238_072_747_940_578_789);
        assert_eq!(mix64(11_011), 8_661_172_894_039_447_344);
        assert_eq!(mix64(GOLDEN_GAMMA), 16_294_208_416_658_607_535);
    }

    #[test]
    fn game_seeds_depend_only_on_seed_and_index() {
        assert_eq!(derive_game_seed(11_011, 0), 16_074_279_617_529_879_324);
        assert_eq!(derive_game_seed(11_011, 1), 5_196_797_165_177_415_115);
        // Nearby base seeds must not produce nearby streams — the failure the
        // golden-ratio stride plus a full-avalanche mixer exists to prevent.
        assert_ne!(derive_game_seed(1, 0), derive_game_seed(2, 0));
    }

    #[test]
    fn sign_collapses_the_three_terminal_values() {
        assert_eq!(sign(1.0), 1);
        assert_eq!(sign(-1.0), -1);
        assert_eq!(sign(0.0), 0);
    }

    /// A tiny run with no net (the searcher falls back to the hand-tuned leaf),
    /// so the whole contract is exercised in under a second.
    fn tiny(shard_idx: u64, shard_count: u64) -> (Stats, String) {
        tiny_with(shard_idx, shard_count, 4, None)
    }

    /// `tiny`, parameterised by game count and resign settings.
    ///
    /// The resign tests deliberately run at a threshold of `0.0` rather than
    /// the production `0.95`: six turns of hand-tuned self-play from the
    /// opening never produces a `|v| > 0.95` root, so a test at the real
    /// threshold would assert on a rule that never fired and would keep passing
    /// if the rule were deleted. `0.0` means "whoever is behind at all", which
    /// exercises exactly the same code on positions a fast test can reach.
    fn tiny_with(
        shard_idx: u64,
        shard_count: u64,
        games: u64,
        resign: Option<ResignConfig>,
    ) -> (Stats, String) {
        let options = Options {
            games,
            seed: 7,
            shard_idx,
            shard_count,
            game: GameConfig {
                sims: 8,
                max_turns: 6,
                resign,
                ..GameConfig::default()
            },
        };
        let mut out: Vec<u8> = Vec::new();
        let stats = generate(&options, None, &mut out).expect("writing to a Vec cannot fail");
        (stats, String::from_utf8(out).expect("rows are valid UTF-8"))
    }

    /// A rule that fires early and often, with no control arm, so a test can
    /// see resignations at all. `control_frac` is overridden where it matters.
    fn eager() -> ResignConfig {
        ResignConfig {
            threshold: 0.0,
            plies: 2,
            control_frac: 0.0,
        }
    }

    /// One game's config, matching `tiny_with`'s.
    fn tiny_game(resign: Option<ResignConfig>) -> GameConfig {
        GameConfig {
            sims: 8,
            max_turns: 6,
            resign,
            ..GameConfig::default()
        }
    }

    #[test]
    fn rows_satisfy_the_selfplay_contract() {
        let (stats, jsonl) = tiny(0, 1);
        assert_eq!(stats.games, 4);
        assert!(stats.rows > 0, "a 4-game run must produce rows");
        assert_contract(&jsonl, 4);
    }

    /// The acceptance gate says the schema must not change when resignation is
    /// on, so the contract check runs against a resign-enabled run too — a
    /// truncated game is still a game, and half of it is still the same rows.
    #[test]
    fn resign_enabled_rows_satisfy_the_same_contract() {
        let rules = ResignConfig {
            control_frac: 0.5,
            ..eager()
        };
        let (stats, jsonl) = tiny_with(0, 1, 4, Some(rules));
        assert!(stats.resigned > 0, "the eager rule must actually fire here");
        assert!(
            stats.rows > 0,
            "a resigned game still emits its opening rows"
        );
        assert_contract(&jsonl, 4);
    }

    fn assert_contract(jsonl: &str, games: usize) {
        let mut per_game: std::collections::HashMap<String, Vec<i8>> = Default::default();
        for line in jsonl.lines() {
            let row: serde_json::Value = serde_json::from_str(line).expect("each line is JSON");
            let object = row.as_object().expect("each row is an object");
            // Key *order* has to be read off the raw line: `serde_json::Value`
            // parses into a sorted map by default, so asserting on its keys
            // would test serde's ordering rather than the emitter's.
            let mut cursor = 0usize;
            for key in ["g", "sym", "ml", "nuo", "nux", "mover", "pi", "pv", "z"] {
                let needle = format!("\"{key}\":");
                let at = line[cursor..]
                    .find(&needle)
                    .unwrap_or_else(|| panic!("{key} missing or out of SelfPlayMcts order"));
                cursor += at + needle.len();
            }

            let sym = object["sym"].as_array().expect("sym is an array");
            assert_eq!(sym.len(), 144);
            assert!(sym.iter().all(|s| (0..8).contains(&s.as_u64().unwrap())));
            assert!((1..=3).contains(&object["ml"].as_u64().unwrap()));
            assert!(object["nuo"].as_u64().unwrap() <= 1);
            assert!(object["nux"].as_u64().unwrap() <= 1);
            assert!((1..=2).contains(&object["mover"].as_u64().unwrap()));

            let pi: Vec<u64> = object["pi"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let pv: Vec<u64> = object["pv"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(pi.len(), pv.len());
            assert!(pi.len() > 1, "forced positions must not be emitted");
            assert!(pv.iter().sum::<u64>() > 0, "an all-zero target is useless");
            assert_eq!(
                pi.iter().collect::<std::collections::HashSet<_>>().len(),
                pi.len(),
                "duplicate action ids alias two targets onto one"
            );
            for id in &pi {
                assert!((*id as usize) < virus_mcts::ACTION_ID_COUNT);
                if *id >= 144 {
                    let rest = *id - 144;
                    assert!(
                        rest / 144 < rest % 144,
                        "pair id {id} must be 144 + min*144 + max"
                    );
                }
                // The mover cannot offer a neutral pair once it has spent its
                // placement — the cross-field invariant validate_rows.py checks.
                if object["nuo"].as_u64().unwrap() == 1 {
                    assert!(*id < 144, "nuo=1 but pi offers a neutral pair");
                }
            }

            let z = object["z"].as_i64().unwrap() as i8;
            assert!((-1..=1).contains(&z));
            per_game
                .entry(object["g"].as_str().unwrap().to_owned())
                .or_default()
                .push(z);
        }

        assert_eq!(per_game.len(), games, "one id per game");
        for (game, zs) in &per_game {
            let first = zs[0];
            assert!(
                zs.iter().all(|z| *z == first),
                "{game}: z must be one per-game absolute label, not per row"
            );
        }
    }

    #[test]
    fn same_seed_and_shard_is_byte_identical() {
        assert_eq!(tiny(0, 1).1, tiny(0, 1).1);
        assert_eq!(tiny(1, 2).1, tiny(1, 2).1);
    }

    /// The load-bearing half of the determinism criterion: shard *count* must
    /// not change which games exist or what they contain, only who plays them.
    #[test]
    fn sharding_repartitions_rather_than_regenerates() {
        let whole = tiny(0, 1).1;
        let halves = format!("{}{}", tiny(0, 2).1, tiny(1, 2).1);

        let sorted = |text: &str| {
            let mut lines: Vec<&str> = text.lines().collect();
            lines.sort_unstable();
            lines.join("\n")
        };
        assert_eq!(sorted(&whole), sorted(&halves));

        // And the split is the modulo one: shard 0 of 2 owns games 0 and 2.
        let games = |text: &str| {
            let mut ids: Vec<String> = text
                .lines()
                .map(|line| {
                    serde_json::from_str::<serde_json::Value>(line).unwrap()["g"]
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            ids.sort();
            ids.dedup();
            ids
        };
        assert_eq!(games(&tiny(0, 2).1), ["sp7-0", "sp7-2"]);
        assert_eq!(games(&tiny(1, 2).1), ["sp7-1", "sp7-3"]);
    }

    /// A capped game keeps its territory verdict. Six turns is far too few to
    /// finish a 12x12 game, so every game in `tiny` is capped — and a run whose
    /// outcomes were all `0` would mean the cap had been turned into a draw.
    #[test]
    fn a_capped_game_is_scored_by_territory_not_called_a_draw() {
        let (stats, _) = tiny(0, 1);
        assert_eq!(stats.capped, 4, "six turns cannot finish a 12x12 game");
        assert!(
            stats.outcomes[0] + stats.outcomes[2] > 0,
            "capped games were all scored 0 — the turn cap became a fake draw"
        );
    }

    /// The boundary the `capped` flag is easiest to get wrong on: a game whose
    /// *last allowed* action ends it leaves the loop on the ceiling condition,
    /// not through the `game_over` break, so a naive flag reports a finished
    /// game as capped.
    #[test]
    fn a_game_ending_on_its_last_allowed_ply_is_not_reported_capped() {
        let generous = GameConfig {
            sims: 8,
            max_turns: DEFAULT_MAX_TURNS,
            ..GameConfig::default()
        };
        let per_turn = u32::from(ACTIONS_PER_TURN);

        // A cap can only land exactly on the final action of a game whose
        // length is a whole number of turns, so find one.
        let (seed, plies) = (0..32u64)
            .map(|game| {
                let report = play_game("probe", derive_game_seed(1, game), None, &generous);
                (game, report)
            })
            .find(|(_, report)| !report.capped && report.plies % per_turn == 0)
            .map(|(game, report)| (game, report.plies))
            .expect("some short game finishes on a turn boundary");

        let exact = GameConfig {
            max_turns: plies / per_turn,
            ..generous
        };
        let report = play_game("probe", derive_game_seed(1, seed), None, &exact);
        assert_eq!(
            report.plies, plies,
            "the tightened cap must land exactly on the final action"
        );
        assert!(
            !report.capped,
            "a game that ended on its last allowed ply is finished, not capped"
        );
    }

    // ---- resignation ----------------------------------------------------

    #[test]
    fn hopeless_is_read_in_the_absolute_frame_per_seat() {
        // The same value means opposite things to the two seats, and a
        // threshold test that forgot that would resign the *winner*.
        assert!(is_hopeless(1, -0.96, 0.95));
        assert!(!is_hopeless(2, -0.96, 0.95));
        assert!(is_hopeless(2, 0.96, 0.95));
        assert!(!is_hopeless(1, 0.96, 0.95));
        // Strictly beyond, so threshold 1.0 is "never" rather than "at every
        // proven-lost root".
        assert!(!is_hopeless(1, -0.95, 0.95));
        assert!(!is_hopeless(2, 0.95, 0.95));
    }

    #[test]
    fn the_hopeless_counter_is_per_seat_and_ignores_the_opponents_plies() {
        let rules = ResignConfig {
            threshold: 0.5,
            plies: 3,
            control_frac: 0.0,
        };
        let mut tracker = ResignTracker::default();
        // -0.9 is hopeless for player 1 and comfortable for player 2, so the
        // two plies in the middle must leave player 1's count alone. A single
        // shared counter would have reset it and never fired.
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(!tracker.observe(2, -0.9, &rules));
        assert!(!tracker.observe(2, -0.9, &rules));
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(tracker.observe(1, -0.9, &rules), "three own hopeless plies");
    }

    #[test]
    fn one_survivable_ply_resets_the_count() {
        let rules = ResignConfig {
            threshold: 0.5,
            plies: 3,
            control_frac: 0.0,
        };
        let mut tracker = ResignTracker::default();
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(!tracker.observe(1, 0.0, &rules), "back to even");
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(!tracker.observe(1, -0.9, &rules));
        assert!(
            tracker.observe(1, -0.9, &rules),
            "the streak restarted at 0"
        );
    }

    #[test]
    fn resign_is_off_by_default_and_inert_when_absent() {
        assert!(
            GameConfig::default().resign.is_none(),
            "a generation that did not ask for resignation must not get it"
        );
        let (stats, baseline) = tiny(0, 1);
        assert_eq!(stats.resigned, 0);
        assert_eq!(stats.control, 0);
        assert_eq!(stats.control_would_resign, 0);
        assert!(stats.false_resign_rate().is_none());
        assert!(stats.projected_search_saving().is_none());
        assert_eq!(tiny_with(0, 1, 4, None).1, baseline);
    }

    #[test]
    fn control_arm_membership_is_a_pure_function_of_seed_and_game() {
        const N: u64 = 4_000;
        let arm = |seed: u64, frac: f64| -> Vec<u64> {
            (0..N).filter(|g| is_control_game(seed, *g, frac)).collect()
        };

        let members = arm(11_011, 0.10);
        let rate = members.len() as f64 / N as f64;
        assert!(
            (rate - 0.10).abs() < 0.02,
            "the control coin drew {rate}, which is not a 10% arm"
        );
        assert_eq!(members, arm(11_011, 0.10), "the draw must be repeatable");
        assert_ne!(
            members,
            arm(11_012, 0.10),
            "a different base seed must give a different arm assignment"
        );
        // The two degenerate ends, which the flag validation allows.
        assert!(arm(11_011, 0.0).is_empty());
        assert_eq!(arm(11_011, 1.0).len() as u64, N);
        // Nested: raising the fraction only ever adds games, so a run at 10%
        // and a run at 20% share their control arm rather than redrawing it.
        let wider = arm(11_011, 0.20);
        assert!(members.iter().all(|g| wider.contains(g)));
    }

    #[test]
    fn a_control_game_is_byte_for_byte_the_resign_off_game() {
        // Everything is control, so the rule is evaluated but never acted on.
        let watching = ResignConfig {
            control_frac: 1.0,
            ..eager()
        };
        for game in 0..8u64 {
            let seed = derive_game_seed(7, game);
            let control = play_game("probe", seed, None, &tiny_game(Some(watching)));
            let off = play_game("probe", seed, None, &tiny_game(None));
            assert!(control.control);
            assert!(control.resigned.is_none(), "a control game never resigns");
            assert_eq!(control.plies, off.plies);
            assert_eq!(control.searches, off.searches);
            assert_eq!(control.z, off.z, "a control game keeps the board verdict");
            assert_eq!(
                control.rows, off.rows,
                "measuring the rule must not change the game being measured"
            );
        }
    }

    #[test]
    fn resign_truncates_rather_than_perturbing() {
        let rules = ResignConfig {
            control_frac: 0.0,
            ..eager()
        };
        let mut seen = 0;
        for game in 0..8u64 {
            let seed = derive_game_seed(7, game);
            let short = play_game("probe", seed, None, &tiny_game(Some(rules)));
            let full = play_game("probe", seed, None, &tiny_game(None));
            let Some(_) = short.resigned else { continue };
            seen += 1;

            assert!(short.plies < full.plies);
            assert!(short.rows.len() <= full.rows.len());
            assert!(!short.capped, "a resigned game did not run out of turns");
            for (truncated, played_out) in short.rows.iter().zip(&full.rows) {
                // Every field but `z` must be identical: the rule decides how
                // long a game is, never what happened in it. `z` is exactly the
                // field it is allowed to change — that is what a false
                // resignation *is*.
                assert_eq!(
                    &Row {
                        z: played_out.z,
                        ..truncated.clone()
                    },
                    played_out,
                    "resignation perturbed the game it was only meant to shorten"
                );
            }
        }
        assert!(seen > 0, "no game resigned — the test proved nothing");
    }

    #[test]
    fn a_resigned_game_is_won_by_the_side_that_did_not_resign() {
        let rules = ResignConfig {
            control_frac: 0.0,
            ..eager()
        };
        let mut disagreed = 0;
        for game in 0..8u64 {
            let seed = derive_game_seed(7, game);
            let short = play_game("probe", seed, None, &tiny_game(Some(rules)));
            let Some(loser) = short.resigned else {
                continue;
            };
            assert!(loser == 1 || loser == 2);
            assert_eq!(
                short.z,
                if loser == 1 { -1 } else { 1 },
                "the resigning side must be recorded as having LOST, in the \
                 absolute frame"
            );
            assert!(
                short.rows.iter().all(|row| row.z == short.z),
                "z is one per-game label"
            );
            disagreed += usize::from(short.z != play_game("probe", seed, None, &tiny_game(None)).z);
        }
        assert!(
            disagreed > 0,
            "every resigned game happened to agree with its played-out result, \
             so this run cannot tell a resignation verdict from a board verdict"
        );
    }

    #[test]
    fn the_control_counterfactual_is_exactly_what_the_resign_arm_would_do() {
        // The measurement's whole validity: what the control arm records as
        // "would have resigned here" has to be where the resign arm actually
        // stops on the same game. If these two ever drift, the false-resign
        // rate is measuring a rule nobody is running.
        let watching = ResignConfig {
            control_frac: 1.0,
            ..eager()
        };
        let acting = ResignConfig {
            control_frac: 0.0,
            ..eager()
        };
        let mut fired = 0;
        for game in 0..8u64 {
            let seed = derive_game_seed(7, game);
            let control = play_game("probe", seed, None, &tiny_game(Some(watching)));
            let resign = play_game("probe", seed, None, &tiny_game(Some(acting)));
            match (control.would_resign, resign.resigned) {
                (Some((side, at)), Some(loser)) => {
                    fired += 1;
                    assert_eq!(side, loser);
                    assert_eq!(at, resign.plies, "the arm stopped at the recorded ply");
                    assert_eq!(
                        resign.searches,
                        at + 1,
                        "the resigning ply's own search still happened"
                    );
                    assert!(at < control.plies);
                }
                (None, None) => {}
                other => panic!("the two arms disagree on game {game}: {other:?}"),
            }
        }
        assert!(fired > 0, "the rule never fired — the test proved nothing");
    }

    #[test]
    fn same_seed_and_shard_is_byte_identical_with_resign_on() {
        let rules = Some(ResignConfig {
            control_frac: 0.5,
            ..eager()
        });
        assert_eq!(
            tiny_with(0, 1, 8, rules).1,
            tiny_with(0, 1, 8, rules).1,
            "resignation must not introduce a second source of randomness"
        );
    }

    #[test]
    fn sharding_repartitions_the_arms_rather_than_redrawing_them() {
        let rules = Some(ResignConfig {
            control_frac: 0.5,
            ..eager()
        });
        let sorted = |text: &str| {
            let mut lines: Vec<&str> = text.lines().collect();
            lines.sort_unstable();
            lines.join("\n")
        };

        let (whole, whole_rows) = tiny_with(0, 1, 8, rules);
        let (first, first_rows) = tiny_with(0, 2, 8, rules);
        let (second, second_rows) = tiny_with(1, 2, 8, rules);

        assert_eq!(
            sorted(&whole_rows),
            sorted(&format!("{first_rows}{second_rows}"))
        );
        assert!(
            whole.control > 0 && whole.control < whole.games,
            "mixed arms"
        );
        // Every measurement counter is a sum over games, so a re-sharded run
        // must report exactly the same totals — otherwise the false-resign
        // rate would depend on how many processes the run was split across.
        assert_eq!(whole.control, first.control + second.control);
        assert_eq!(whole.resigned, first.resigned + second.resigned);
        assert_eq!(
            whole.control_would_resign,
            first.control_would_resign + second.control_would_resign
        );
        assert_eq!(
            whole.control_false_resign,
            first.control_false_resign + second.control_false_resign
        );
        assert_eq!(
            whole.control_searches_saved,
            first.control_searches_saved + second.control_searches_saved
        );
        assert_eq!(whole.searches, first.searches + second.searches);
    }

    #[test]
    fn a_run_with_a_control_arm_measures_the_rate_the_bar_is_written_against() {
        let rules = ResignConfig {
            control_frac: 0.5,
            ..eager()
        };
        let (stats, _) = tiny_with(0, 1, 16, Some(rules));
        assert_eq!(stats.games, 16);
        assert!(stats.control > 0 && stats.control < stats.games);
        assert!(stats.resigned > 0, "the resign arm resigned nothing");
        assert!(stats.resigned <= stats.games - stats.control);
        assert!(stats.control_would_resign > 0);
        assert!(stats.control_false_resign <= stats.control_would_resign);

        let rate = stats.false_resign_rate().expect("the rule fired");
        assert!((0.0..=1.0).contains(&rate));
        let saving = stats.projected_search_saving().expect("a control arm ran");
        assert!(
            (0.0..1.0).contains(&saving) && saving > 0.0,
            "a rule that resigns games must save searches, got {saving}"
        );
        assert!(stats.control_plies_saved >= stats.control_searches_saved);
    }

    #[test]
    fn a_would_be_resigner_that_only_drew_still_counts_as_a_false_resign() {
        // Constructed rather than played: the bar is "would NOT have lost", and
        // the difference between that and "would have won" is a handful of
        // games at the margin of a 5% threshold.
        let stats = Stats {
            control_would_resign: 4,
            control_false_resign: 1,
            control_searches: 400,
            control_searches_saved: 100,
            ..Stats::default()
        };
        assert_eq!(stats.false_resign_rate(), Some(0.25));
        assert_eq!(stats.projected_search_saving(), Some(0.25));
        assert_eq!(Stats::default().false_resign_rate(), None);
        assert_eq!(Stats::default().projected_search_saving(), None);
    }
}
