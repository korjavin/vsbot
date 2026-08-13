# Deploying vsbot

How the bot is containerized and how it runs against the public server at
`wss://vs.wandergeek.org/ws`.

## What is actually deployed today

**A protocol soak test, not a strong bot.** The only engine wired into the
binary is `GREEDY` — the deliberately weak reference engine that exists so the
wire protocol can be exercised end to end. `virus-search` (alpha-beta) and
`virus-mcts` (PUCT + policy/value net) are still stubs, and selecting them is a
*hard startup failure* rather than a silent fallback, so a deployment can never
believe it is running the champion while playing at random.

So this stack proves: TLS and the WebSocket handshake, `welcome`/registration,
reconnect with backoff, turn handling, and move legality. It proves nothing
about playing strength. Flip `SEARCH` once the engines merge.

## Environment knobs

All configuration is environment variables, read in exactly one place
(`crates/vsbot/src/main.rs`). Unset and empty mean the same thing; an
unparseable value **fails startup** instead of quietly falling back.

| Variable | Default (image) | Meaning |
| --- | --- | --- |
| `BACKEND_URL` | `ws://localhost:8080/ws` | WebSocket endpoint. The binary appends `?bot=true` and `&namePrefix=…` itself — do not put them in the URL. |
| `BOT_NAME_PREFIX` | *(empty)* | Prefixed onto the server-assigned name. Compose sets `SuperiorBot`, which the lobby shows as e.g. `SuperiorBot Bot 2127`. |
| `MOVE_MILLIS` | `1000` | Wall-clock budget per action. The owner's UX bound is 10–15 s for a 3-action turn, so stay well under ~3000. |
| `SEARCH` | `GREEDY` | `GREEDY` \| `ALPHABETA` \| `MCTS`. The latter two abort startup until their crates land. |
| `CHALLENGER` | `false` | Whether the bot initiates games on a timer. **Keep this `false` in production** (see below). |
| `CHALLENGER_INTERVAL_SECS` | `300` | Challenger period; first tick is jittered. Irrelevant while `CHALLENGER=false`. |

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
vsbot 0.1.0 starting: url=wss://vs.wandergeek.org/ws search=GREEDY move_budget=1s challenger=false
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
MOVE_MILLIS=1500
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

Driving the Portainer *API* from CI (a redeploy webhook, as `nnue-trainer`
does) needs a webhook URL stored as a repository secret. That is deliberately
not set up here — no credentials were available — so redeploys are manual.

## Promote and roll back

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
