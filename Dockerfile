# syntax=docker/dockerfile:1
#
# vsbot deployment image.
#
# Two stages: a full Rust toolchain compiles the workspace, and a slim Debian
# runtime carries only the binary. No OpenSSL is needed at runtime — the
# WebSocket client is built on `tokio-tungstenite` with
# `rustls-tls-webpki-roots` (see the workspace Cargo.toml), so the trust anchors
# are *compiled into the binary*. That is also why the runtime stage installs no
# `ca-certificates`: `wss://` works with an empty image.

# --------------------------------------------------------------- build stage
FROM rust:1-slim-trixie AS builder

WORKDIR /build

# Dependency pre-build. Copying only the manifests and stub sources means the
# expensive layer (ring, rustls, tokio, serde — everything third-party) is
# rebuilt only when a Cargo.toml or Cargo.lock changes, not on every source
# edit. Each workspace member is a library except `vsbot`, which is the bin.
COPY Cargo.toml Cargo.lock ./
COPY crates/virus-core/Cargo.toml   crates/virus-core/Cargo.toml
COPY crates/virus-eval/Cargo.toml   crates/virus-eval/Cargo.toml
COPY crates/virus-search/Cargo.toml crates/virus-search/Cargo.toml
COPY crates/virus-mcts/Cargo.toml   crates/virus-mcts/Cargo.toml
COPY crates/virus-proto/Cargo.toml  crates/virus-proto/Cargo.toml
COPY crates/virus-arena/Cargo.toml  crates/virus-arena/Cargo.toml
COPY crates/vsbot/Cargo.toml        crates/vsbot/Cargo.toml
RUN set -eux; \
    for lib in virus-core virus-eval virus-search virus-mcts virus-proto virus-arena; do \
        mkdir -p "crates/$lib/src"; \
        : > "crates/$lib/src/lib.rs"; \
    done; \
    mkdir -p crates/vsbot/src; \
    printf 'fn main() {}\n' > crates/vsbot/src/main.rs; \
    cargo build --release -p vsbot --locked

# Real sources. `touch` is load-bearing: Docker's COPY preserves the source
# mtimes, which can be older than the stub build above, and cargo would then
# consider the crates up to date and ship the empty stub binary.
COPY crates ./crates
RUN set -eux; \
    find /build/crates -name '*.rs' -exec touch {} +; \
    cargo build --release -p vsbot --locked; \
    strip /build/target/release/vsbot

# ------------------------------------------------------------- runtime stage
FROM debian:trixie-slim AS runtime

# The binary needs nothing beyond glibc, which the base image already has.
# Run unprivileged: the bot only opens one outbound TCP connection.
RUN set -eux; \
    useradd --system --create-home --home-dir /home/vsbot --shell /usr/sbin/nologin vsbot

COPY --from=builder /build/target/release/vsbot /usr/local/bin/vsbot

# Trained artifacts, baked in so a future MCTS deployment does not have to
# mount a volume. NOTE (2026-08-13): the `vsbot` binary has **no config surface
# for these yet** — it reads BACKEND_URL / BOT_NAME_PREFIX / MOVE_MILLIS /
# SEARCH / CHALLENGER / CHALLENGER_INTERVAL_SECS and nothing else. These files
# are staged here for when `virus-mcts` lands and gains an artifact-path knob.
# See DEPLOY.md; do not assume setting some env var activates them today.
COPY artifacts /opt/vsbot/artifacts

USER vsbot
WORKDIR /home/vsbot

# Defaults are deliberately the safe ones: the local dev backend, accept-only
# (no challenger), and the reference greedy engine. docker-compose.yml supplies
# the production values.
ENV BACKEND_URL=ws://localhost:8080/ws \
    SEARCH=GREEDY \
    MOVE_MILLIS=1000 \
    CHALLENGER=false

ENTRYPOINT ["/usr/local/bin/vsbot"]
