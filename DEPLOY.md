# Deploying vsbot

How the bot is containerized and how it runs against the public server at
`wss://vs.wandergeek.org/ws`.

## What is actually deployed today

**The gen-5 MCTS champion.** `SEARCH=MCTS` is the default: PUCT over the
conv policy/value net (`MCTS_ARTIFACT`, baked into the image), batched SIMD
inference, driven by the intra-turn time allocator at 10 s per 3-action turn.
A bad or missing artifact is a *hard startup failure*, and a per-game domain
mismatch (>2 players, non-12×12) falls back to `GREEDY` with a `WARNING` line
on every change of verdict — a deployment can never silently believe it is
running the champion while playing at random. Strength claims still come only
from ≥400-game gauntlets (`docs/benchmarks.md`), never from live play.

## Environment knobs

All configuration is environment variables, read in exactly one place
(`crates/vsbot/src/main.rs`). Unset and empty mean the same thing; an
unparseable value **fails startup** instead of quietly falling back.

| Variable | Default (image) | Meaning |
| --- | --- | --- |
| `BACKEND_URL` | `ws://localhost:8080/ws` | WebSocket endpoint. The binary appends `?bot=true` and `&namePrefix=…` itself — do not put them in the URL. |
| `BOT_NAME_PREFIX` | *(empty)* | Prefixed onto the server-assigned name. Compose sets `SuperiorBot`, which the lobby shows as e.g. `SuperiorBot Bot 2127`. |
| `VSBOT_TURN_MILLIS` | `10000` | Wall-clock budget for a whole **3-action turn**, split 50/30/20 across the turn's actions. Set to 10 s by owner verdict after live play (PR #16); the server's 120 s per-action timer is a failsafe, never a budget. |
| `VSBOT_MOVE_MILLIS` | *(unset)* | Per-action override. Setting it **disables the intra-turn allocator** — every action gets exactly this, nothing is banked or released. Use it for gauntlets and A/Bs, not for live play. |
| `MOVE_MILLIS` | *(unset)* | The pre-allocator spelling of `VSBOT_MOVE_MILLIS`, still honoured so an existing `.env` keeps meaning what it meant. If both are set, `VSBOT_MOVE_MILLIS` wins. |
| `VSBOT_EARLY_STOP` | `true` | Stop as soon as the visit leader cannot be overtaken in what is left of the budget, and give the remainder to the next action. |
| `VSBOT_EXTENSION` | `true` | Run past an action's target toward its ceiling while the root is unstable. The ceiling is capped by what remains of the turn, so extensions can never break the turn bound. |
| `VSBOT_PONDER` | `false` | Search the opponent's positions during their turn, re-rooting into the matched child on each of their actions. **Off pending bug `vsbot-gei`** (2026-08-13 canary found a quality regression); when fixed, it re-ships through the automated canary path (docs/CANARY.md). |
| `VSBOT_PONDER_SECS` | `30` | Cap on one pondering step. Bounds CPU and tree memory on a host shared with the nightly trainer window. |
| `SEARCH` | `MCTS` | `GREEDY` \| `ALPHABETA` \| `MCTS`. `MCTS` is the default (compose and image agree). `ALPHABETA` aborts startup — the crate is merged but not yet wired into the binary. |
| `CHALLENGER` | `false` | Whether the bot initiates games on a timer. **Keep this `false` in production** (see below). |
| `CHALLENGER_INTERVAL_SECS` | `300` | Challenger period; first tick is jittered. Irrelevant while `CHALLENGER=false`. |
| `VSBOT_EXPLORE_EPS` | `0` *(off)* | **Measurement harness only — never set this in production.** Probability that an opening action is replaced by a uniformly random legal one, i.e. the bot playing *deliberately worse* so a cross-play run's games differ from each other (bd `vsbot-t3q.2`, `crates/vsbot/src/explore.rs`). At `0` no RNG is interposed at all. Refused together with `VSBOT_PONDER=true`. |
| `VSBOT_EXPLORE_TURNS` | `8` | Length of that window, counted in the bot's **own** turns. Inert while `VSBOT_EXPLORE_EPS=0`. |
| `VSBOT_EXPLORE_SEED` | `1` | Base seed for the per-game exploration stream. Inert while `VSBOT_EXPLORE_EPS=0`. |

### Budget profiles

| Profile | Environment | Why |
| --- | --- | --- |
| **Deployed default** | *(nothing)* | 12 s/turn ≈ 6 s / 3.6 s / 2.4 s. Inside the owner's UX bound with margin for WS round-trip and snapshot revalidation. |
| **Owner canary** | `VSBOT_TURN_MILLIS=15000`, `VSBOT_PONDER=true` | Top of the bound plus pondering. One candidate per canary; the owner's verdict promotes it. |
| **Bot gauntlets** | `VSBOT_MOVE_MILLIS=1000` | Gauntlets and the RL gate stay at fixed time per action so results stay comparable. |
| **Pre-S2 parity** | `VSBOT_MOVE_MILLIS=1000`, `VSBOT_EARLY_STOP=false`, `VSBOT_EXTENSION=false` | Exactly what shipped before the time manager, for an A/B. |

Two counters to watch in the logs (`Counters` in `virus-proto`): `fallback_actions`
must stay **zero** — a non-zero value means an action blew through its ceiling
and was answered with the pre-selected fallback instead of a searched move — and
`illegal_moves` must stay zero, as always.

### Upgrading past the time manager

Before the allocator, the image hard-coded `MOVE_MILLIS=1000` and every action
got a second. Now the image sets `VSBOT_TURN_MILLIS=10000` and no per-action
override, so a plain `docker compose pull && up -d` **changes the bot's pace**:
one 10 s turn instead of three 1 s actions. That is the intended S2 behaviour and
it is inside the owner's UX bound — but it is a behaviour change, so canary it
rather than rolling it straight onto the default identity.

An operator who had `MOVE_MILLIS=…` in their `.env` keeps exactly what they had:
compose still forwards `MOVE_MILLIS` and `VSBOT_MOVE_MILLIS`, and either one
disables the allocator. To go back to the old pace deliberately, put
`VSBOT_MOVE_MILLIS=1000` in `.env`.

Compose-only knobs (consumed by `docker-compose.yml`, not by the binary):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSBOT_TAG` | `latest` | Which published image tag to run. Set it to pin or roll back. |

### Why `CHALLENGER` stays off

Challenger mode makes the bot initiate games on a timer. Against the *public*
server that means pestering real humans with a greedy bot. The deployment is
accept-only: it sits in the lobby and plays whoever challenges it. Turn it on
only against a private/staging backend.

### Trained artifacts

`artifacts/` (`mcts_champion.json`, `ordering_policy.json`) is baked into the
image at **`/opt/vsbot/artifacts`** so a future MCTS deployment needs no volume
mount.

Be clear about what that does today: **nothing.** The binary has no config
surface for artifact paths — the table above is the complete set of variables it
reads. The files are staged for when `virus-mcts` merges and grows a path knob;
at that point the knob should default to `/opt/vsbot/artifacts` and this section
should say so. Do not assume some env var activates them now.

## Local runbook (this host)

The image is not on GHCR until the first master merge, so the first bring-up
builds from source. `--build` also tags the result as the `image:` value, so
compose finds it afterwards.

```bash
cd /home/devbox/Project/vsbot
docker compose up -d --build          # first time, or to deploy local changes
docker compose logs -f vsbot          # watch it register
```

A healthy start looks like:

```
vsbot 0.1.0 starting: url=wss://vs.wandergeek.org/ws search=MCTS challenger=false
vsbot: budget=10000ms/turn split 50/30/20 across up to 3 actions (early-stop=true, extension=true) ponder=off
vsbot: engine=MCTS artifact=/opt/vsbot/artifacts/mcts_champion.json …
[vsbot] connected to wss://vs.wandergeek.org/ws?bot=true&namePrefix=SuperiorBot
[SuperiorBot Bot 2127] registered as SuperiorBot Bot 2127 (d6c2f97a-…)
```

Once the image exists on GHCR, the ordinary deploy is a pull:

```bash
docker compose pull && docker compose up -d
```

Overrides go in a `.env` file next to `docker-compose.yml` (it is gitignored):

```
BOT_NAME_PREFIX=SuperiorBot
VSBOT_TURN_MILLIS=10000
VSBOT_PONDER=false
VSBOT_TAG=latest
```

Other operations:

```bash
docker compose ps                     # is it up
docker compose restart vsbot          # bounce it
docker compose down                   # stop and remove
docker stats --no-stream vsbot        # against the 2 CPU / 512 MB limits
```

### Checking it is behaving

The bot logs to stderr, prefixed with its lobby name. Two things worth
grepping after a game:

```bash
docker compose logs vsbot | grep -i 'illegal'      # must stay empty
docker compose logs vsbot | grep -iE 'connect|registered|game'
```

An illegal-move forfeit is phrased by the server as
`Defeated by illegal move: …` and is counted internally; any hit here is a real
bug, never a tuning issue.

## Portainer

Portainer CE on this host manages the local Docker endpoint, so the compose
stack above and Portainer are two views of the same thing. Either path works;
pick one and stay with it, because a stack created by the CLI and a stack
created in Portainer are separate objects with separate labels.

**Path A — CLI (what is running now).** `docker compose up -d` as above.
Portainer will show the containers under its Containers view. Simplest, and the
one the acceptance run used.

**Path B — import as a Portainer stack.** Stacks → Add stack → either paste
`docker-compose.yml` (Web editor) or point it at this repository (Repository:
`https://github.com/korjavin/vsbot`, compose path `docker-compose.yml`). Set
`BOT_NAME_PREFIX` and `VSBOT_TAG` as stack environment variables. Note that the
`build:` key only works if Portainer can reach the build context — for a
Repository-backed stack it can; for a pulled image, drop `build:` or set
`VSBOT_TAG` to a published tag so the image is fetched rather than built.

Both paths above are about *this* host, where the source tree is present and a
stack can be built. A Portainer somewhere else should use the `deploy` branch
instead — next section.

## Hosting elsewhere via Portainer

For a Portainer that is **not** on this build host, the repository publishes a
CI-owned `deploy` branch: on every master push, `.github/workflows/docker.yml`
builds and pushes the image, renders `deploy/docker-compose.yml` with the
`image:` pinned to that build's immutable `sha-<commit>` tag, force-pushes the
result to `deploy` as the repository-root `docker-compose.yml`, and then fires
any configured Portainer redeploy webhooks.

```
master push ─► build+push ghcr.io/korjavin/vsbot:{latest,sha-<commit>}
            ─► render deploy/docker-compose.yml with image: …:sha-<commit>
            ─► git push --force origin deploy
            ─► POST $PORTAINER_WEBHOOK_URL…      (optional, silent when unset)
```

So the deploy branch always answers "what is deployed?" with a commit, not with
a moving `:latest`. Nothing on that branch is hand-written — **never commit to
`deploy`; it is rewritten on every master push.**

### 0. One-time: make the GHCR package public

> **Do this before creating the stack.** The `vsbot` container package was first
> published while the repository was private, so the package is still
> **private** even though the repository is public now — GitHub does not flip
> package visibility when the repo's changes, and there is no REST API for it
> either. Verified on 2026-08-13: `docker pull ghcr.io/korjavin/vsbot:latest`
> after `docker logout ghcr.io` fails with `unauthorized`, and even the
> anonymous registry token endpoint refuses (`UNAUTHORIZED`).

The fix is a one-time click, and only the repository owner can do it:

1. <https://github.com/users/korjavin/packages/container/package/vsbot>
   (or github.com → your profile → **Packages** → **vsbot**)
2. **Package settings** (right-hand side)
3. **Danger Zone → Change visibility → Public**, confirm with the package name.

Verify from any machine with no credentials:

```bash
docker logout ghcr.io
docker pull ghcr.io/korjavin/vsbot:latest     # must succeed
```

**Fallback if it stays private** (or if you would rather not publish the
image): give Portainer a registry credential instead. In Portainer,
**Registries → Add registry → Custom registry**, URL `ghcr.io`, username your
GitHub username, password a **classic** personal access token with the
`read:packages` scope. Then, when creating the stack, Portainer will use that
registry's credentials for the pull. The compose file needs no change either
way — it is the same image reference. On a plain Docker host without Portainer
the equivalent is `docker login ghcr.io -u <user> -p <PAT>` once.

### 1. Create the stack from the deploy branch

Portainer → **Stacks** → **Add stack** → **Repository**:

| Field | Value |
| --- | --- |
| Repository URL | `https://github.com/korjavin/vsbot` |
| Repository reference | `refs/heads/deploy` |
| Compose path | `docker-compose.yml` |
| Authentication | off (public repository) |
| GitOps updates | optional — polling works, but the webhook below is instant |

Do **not** point it at `master`: that compose has a `build:` key and a `:latest`
tag, i.e. it asks a remote host to compile the Rust workspace.

### 2. Environment variables

The stack works with an empty environment — every knob has a production default
baked into the compose file. Set only what you want to differ (Portainer's
stack **Environment variables** table):

| Variable | Default on the deploy branch | When to set it |
| --- | --- | --- |
| `BACKEND_URL` | `wss://vs.wandergeek.org/ws` | Pointing at a private/staging backend. |
| `BOT_NAME_PREFIX` | `SuperiorBot` | **Set this if a second installation exists** — two hosts under the same prefix are indistinguishable in the lobby. |
| `VSBOT_TURN_MILLIS` | `10000` | Slower/faster host, or a canary at the top of the UX bound. |
| `SEARCH` | `MCTS` | Only to fall back to `GREEDY`. `ALPHABETA` aborts startup. |
| `VSBOT_PONDER` | `false` | Off pending bd `vsbot-gei`; canary it, do not just turn it on. |
| `CHALLENGER` | `false` | Keep false against the public server (see above). |
| `VSBOT_CONTAINER_NAME` | `vsbot` | Only if another container on that host already owns the name. |
| `VSBOT_CPUS` / `VSBOT_MEMORY` | `2.0` / `512M` | Smaller or larger host. |

The full meaning of each is the table at the top of this file — the deploy
compose forwards the same set the local one does.

### 3. Wire the redeploy webhook (optional)

In the stack's page, **Webhooks** → enable → copy the URL, then add it as a
repository secret:

- <https://github.com/korjavin/vsbot/settings/secrets/actions> → **New repository secret**
- Name: `PORTAINER_WEBHOOK_URL`, value: the copied URL.

Several installations: either put **several URLs in that one secret**, separated
by commas or whitespace, or use `PORTAINER_WEBHOOK_URL_2` and
`PORTAINER_WEBHOOK_URL_3`. All of them are optional — with none set the workflow
says so and moves on, and the deploy branch still updates, so a Portainer with
GitOps polling enabled will catch up on its own schedule. A webhook that
answers non-2xx is reported as a warning on the run (and in the job summary)
but does not fail master: the image and the branch are already correct at that
point. URLs are never printed.

### 4. Promoting, rolling back, and checking what is live

- **Promote:** merge to master. The deploy branch moves, the webhook fires,
  Portainer pulls the new pinned tag. Nothing to do by hand.
- **Roll back:** in the Portainer stack editor, change the one `image:` line to
  the previous `sha-<commit>` tag and redeploy — the whole stack state is that
  one line. Note the next master push rewrites the deploy branch and a
  webhook/GitOps redeploy will roll you forward again, so a rollback that must
  survive is a `git revert` on master, not an edit in Portainer.
- **Which commit is live:** `git ls-remote origin refs/heads/deploy` and read
  the compose on that branch, or read the `image:` in Portainer's stack editor.

```bash
# what the deploy branch currently pins, without cloning it
git fetch origin deploy
git show origin/deploy:docker-compose.yml | grep '^\s*image:'
```

## Promote and roll back

> **Canary first — no net, budget or behaviour change becomes the default
> without an owner canary.** The standing procedure is **[`docs/CANARY.md`](docs/CANARY.md)**:
> the gate ledger a candidate needs before it may be deployed at all, how to run
> it under its own lobby name (`BOT_NAME_PREFIX=Canary-<candidate>`) alongside
> the incumbent, what the owner plays, where the verdict is recorded (the
> candidate's bead), the `artifacts/champions/gen-N.json` history convention, and
> the rollback drill. This section is only the tag mechanics that runbook calls
> into — reaching for it directly means skipping the owner's verdict, which is
> the one thing that promotes anything (`superiority.md` §4 Gate C).

This section is the **local** stack's tag mechanics. A remote Portainer does
not use `VSBOT_TAG` at all — its tag is pinned in the deploy branch's compose
(see "Hosting elsewhere via Portainer" above).

Every master merge publishes two tags from `.github/workflows/docker.yml`:

- `ghcr.io/korjavin/vsbot:latest` — moves with master.
- `ghcr.io/korjavin/vsbot:sha-<full-commit-sha>` — immutable.

Promote (take whatever master has):

```bash
VSBOT_TAG=latest docker compose pull && VSBOT_TAG=latest docker compose up -d
```

Pin or roll back to a known-good commit — put `VSBOT_TAG` in `.env` so it
survives the next `docker compose up`:

```bash
echo 'VSBOT_TAG=sha-9602b2fdeadbeef…' >> .env
docker compose pull && docker compose up -d
docker compose logs -f vsbot          # confirm the banner, then walk away
```

To find candidate tags: `gh api /users/korjavin/packages/container/vsbot/versions`
or the package page on GHCR.

Because the running container is stateless — no volumes, no ports, all config in
env — rollback is just "run the other tag". There is nothing to migrate back.

## Image notes

- Base: `debian:trixie-slim`; builder `rust:1-slim-trixie`. ~123 MB total, of
  which ~87 MB is the Debian base, ~2 MB the stripped binary, ~2 MB artifacts.
- **No `ca-certificates` and no OpenSSL.** TLS is rustls with
  `webpki-roots`, so the trust anchors are compiled into the binary. This is
  why the runtime stage installs nothing at all.
- rustls 0.23 needs its crypto provider chosen explicitly — `tokio-tungstenite`
  0.30 picks none — so `main` installs the `ring` provider before anything
  dials. Without it `wss://` panics on first handshake, which is exactly how it
  failed the first time this image was pointed at production.
- Runs as the unprivileged `vsbot` user, `no-new-privileges`, no listening
  ports. The bot only opens one outbound TCP connection.
- Smaller images are possible (distroless `cc`, or a static musl build → a few
  MB) at the cost of having no shell to debug a live bot with. Not worth it for
  a single container on this host.
