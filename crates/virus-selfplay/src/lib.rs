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

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

use std::io::{self, Write};
use std::time::Instant;

use serde::Serialize;
use virus_core::{Player, State, ACTIONS_PER_TURN};
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
    /// Absolute-frame outcome.
    pub z: i8,
    /// Whether the game stopped at the turn cap rather than finishing.
    pub capped: bool,
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
/// When the loop ends — naturally, at the turn cap, or at a stuck root — the
/// final state is scored once with [`terminal_value_abs`] and that single value
/// is stamped onto every pending row. One flip point, in the trainer; see the
/// module docs.
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
    let mut capped = true;

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

        // A terminal or stuck root: `game_over()` above did not catch it, so
        // score what we have rather than emitting a row with an empty mask.
        if searcher.root_actions().is_empty() {
            capped = false;
            break;
        }
        if searcher.root_actions().len() > 1 {
            pending.push(row(game_id, &state, &searcher));
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

    let z = sign(terminal_value_abs(&state));
    for row in &mut pending {
        row.z = z;
    }
    GameReport {
        rows: pending,
        plies: ply,
        z,
        capped,
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
    /// Wall clock.
    pub seconds: f64,
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
        let options = Options {
            games: 4,
            seed: 7,
            shard_idx,
            shard_count,
            game: GameConfig {
                sims: 8,
                max_turns: 6,
                ..GameConfig::default()
            },
        };
        let mut out: Vec<u8> = Vec::new();
        let stats = generate(&options, None, &mut out).expect("writing to a Vec cannot fail");
        (stats, String::from_utf8(out).expect("rows are valid UTF-8"))
    }

    #[test]
    fn rows_satisfy_the_selfplay_contract() {
        let (stats, jsonl) = tiny(0, 1);
        assert_eq!(stats.games, 4);
        assert!(stats.rows > 0, "a 4-game run must produce rows");

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

        assert_eq!(per_game.len(), 4, "four distinct game ids");
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
}
