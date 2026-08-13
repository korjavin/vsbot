//! The bot binary.
//!
//! **Stub.** Wiring lands with bead `vsbot-kw5`. Per CLAUDE.md, env-var config
//! is read here and *only* here, then passed down as plain structs — no crate
//! below this one reads the environment.

use virus_core::State;

fn main() {
    let state = State::new(12, 12, 2).expect("12x12 two-player board is valid");
    println!(
        "vsbot {} — rules engine online: {}x{}, {} players, {} legal actions at the start",
        env!("CARGO_PKG_VERSION"),
        state.rows(),
        state.cols(),
        state.players(),
        state.legal_actions().len(),
    );
    println!("search, eval, mcts, proto and arena are stubs; see ARCHITECTURE.md");
}
