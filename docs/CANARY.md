# Canary promotion runbook

**Standing procedure. It applies to every production candidate — every net, every
budget change, every behaviour flag — and nothing becomes the default without
completing it.**

The rule it exists to enforce (`docs/plans/superiority.md` §4 Gate C, and the
canary doc `~/Project/virusgame/docs/nnue-canary.md:9-16`): **the owner is the
only promotion judge.** Gauntlet gates are sanity floors, never strength proof.

This runbook is the "how". `DEPLOY.md` is the "what the knobs are" — every
environment variable named here is defined there and read in exactly one place
(`crates/vsbot/src/lib.rs`).

---

## One-screen checklist

Copy this into the candidate's bead and tick it off.

```
CANDIDATE: ____________________  BEAD: vsbot-____
[ ]  0. Gate A recorded   — >=0.55 pooled over >=400 games vs the current champion
[ ]  1. Gate B recorded   — no regression vs ab-enhanced AND vs the Go bot
[ ]     Both scores written to docs/benchmarks.md + the bead. FLOORS, NOT PROOF.
[ ]  2. Incumbent baseline captured (image digest + resolved env) — this is the
[ ]     rollback target. Write the sha- tag into the bead now, not later.
[ ]  3. Candidate published as an immutable tag (ghcr.io/korjavin/vsbot:sha-<sha>)
[ ]     or the candidate artifact staged for a read-only bind mount.
[ ]  4. docker-compose.canary.yml written: BOT_NAME_PREFIX=Canary-<candidate>,
[ ]     container_name: vsbot-canary, no `build:` key.
[ ]  5. Canary up ALONGSIDE the incumbent; banner checked; illegal_moves and
[ ]     fallback_actions both zero.
[ ]  6. Owner told the lobby name. Owner plays >= 5 completed games, including
[ ]     the turtle probe and the gifting probe, plus 1 control game vs the incumbent.
[ ]  7. Verdict appended to the bead: date + PROMOTE|REJECT|ITERATE + impressions.
[ ]  8. On PROMOTE only: artifact -> artifacts/champions/gen-N.json + copied over
[ ]     artifacts/mcts_champion.json, committed, image rebuilt, VSBOT_TAG re-pinned.
[ ]  9. Canary container removed (`docker compose ... rm -sf vsbot-canary`).
[ ] 10. Rollback tag from step 2 still recorded in the bead after promotion.
```

**One candidate per canary.** Never change the net and the budget in the same
canary — the owner's verdict then means nothing about either
(canary doc:89-95, "one net per branch so the owner can A/B them").

---

## Why this exists (the Goodhart lesson, cited)

`vs-ai2.52` dominated every scripted gate it was measured against, then lost
three straight games to the owner by base-turtling
(`~/Project/virusgame/docs/nnue-canary.md:9-12`; bd memory
`vs-ai2-goodhart-scripted-gates`). Gate fitness did **not** transfer to strength
against the actual target opponent.

So: gauntlet numbers can only *stop* a promotion, never cause one. A candidate
that clears Gate A and Gate B has earned a canary slot — nothing more.

---

## 1. Before the canary: the gate ledger

A candidate does not get deployed, even under a canary name, until both floors
are measured and written down.

### Gate A — beat the current champion

`>= 0.55` pooled `(W + 0.5·D)/N` over `N >= 400` games at fixed sims,
colour-paired, per-instance seed ranges spaced >= 1000
(`docs/plans/superiority.md` §4).

> **Harness status, 2026-08-13:** `virus-arena` shares one loaded net across all
> games and threads, so a *net-vs-net* run is refused rather than silently
> playing one artifact against itself — see `docs/benchmarks.md`, "What is not
> measured yet". Until that follow-up bead lands, Gate A for an RL candidate
> comes from the nightly generation gate in `nnue-trainer`
> (`scripts/mcts_selfplay_gen.sh`, same pooled arithmetic), and the candidate's
> bead cites the generation report. Do not fabricate a local number.

### Gate B — no regression against fixed external opponents

`>= 400` games at 1 s/action fixed time against **each** of:

1. the Rust hand-tuned enhanced alpha-beta bar (`ab-enhanced`),
2. the Go bot (cross-play, refereed by a real Go server the script builds and
   spawns locally).

Rule: score `>=` the current champion's recorded score against the same opponent
minus 2·SE (about 5 points at 400 games). Never regress.

```bash
cargo build --release -p virus-arena

# B(i) — candidate vs the hand-tuned bar, equal wall clock.
./target/release/arena \
    --a mcts:artifacts/champions/gen-6.json --b ab-enhanced \
    --budget ms:1000 --games 400 --seed 20260813 --threads 4

# B(ii) — candidate vs the Go bot, refereed by a locally spawned Go server.
cargo build --release -p vsbot
MCTS_ARTIFACT=$PWD/artifacts/champions/gen-6.json \
  python3 crates/virus-arena/crossplay/crossplay.py --games 400 --search MCTS
```

The script has no artifact flag: it inherits the environment and launches
`vsbot` from a temp working directory, so `MCTS_ARTIFACT` must be **absolute**
here or the relative default (`artifacts/mcts_champion.json`) fails to resolve
and startup aborts. It also sets `MOVE_MILLIS` itself (default 1000), which
disables the intra-turn allocator — correct for a fixed-time gate, and the
reason Gate B numbers are not comparable to live play.

Cross-play seats `vsbot` at P1 in every game (the server seats the challenger
there and only `vsbot` challenges), so it carries an uncancelled first-mover
bias and the harness labels it `INFORMATIONAL ONLY`. Compare candidate to
champion on the *same* biased axis — that comparison is still valid — and read
`docs/benchmarks.md` §3 before quoting the magnitude of either.

### Where the numbers live

- **`docs/benchmarks.md`** — one section per gate run, with the exact command
  that reproduces it and the harness's own verdict line (`indicative` /
  `INFORMATIONAL ONLY` / gate-eligible). Nothing under 400 games may gate.
- **The candidate's bead** — the same numbers, plus the artifact's provenance
  (which generation, which trainer run) so the ledger survives a doc rewrite.

Both are **floors**. Write the Goodhart sentence next to them so no future
reader mistakes the ledger for a promotion:

> Gate A/B are sanity floors, not a strength proof. Only the owner's canary
> verdict promotes (superiority.md §4 Gate C; nnue-canary.md:9-16).

---

## 2. Capture the rollback target before you touch anything

```bash
cd /home/devbox/Project/vsbot
docker compose config                                   # resolved env of the incumbent
docker inspect --format '{{.Config.Image}}' vsbot       # the tag it is running
docker inspect --format '{{index .RepoDigests 0}}' \
    ghcr.io/korjavin/vsbot:latest                       # the immutable digest behind :latest
```

Write the resulting `sha-<full-commit-sha>` tag into the candidate's bead now.
`:latest` moves with master, so after the promotion merge it no longer points at
the incumbent — the rollback target must be recorded *before* it moves.

---

## 3. Deploy the canary alongside the incumbent

The canary is a **second service in the same compose project**, brought up by
name. The incumbent keeps playing untouched; the owner sees two bots in the
lobby and can A/B them in one session.

Two facts from `docker-compose.yml` that shape everything below:

- the project name is pinned (`name: vsbot`) and the incumbent's container name
  is pinned (`container_name: vsbot`), so the canary **must** declare its own
  `container_name` or the two collide;
- the stack has **no volumes** — artifacts are baked into the image at
  `/opt/vsbot/artifacts` (`Dockerfile`: `COPY artifacts /opt/vsbot/artifacts`),
  and `MCTS_ARTIFACT` defaults to `artifacts/mcts_champion.json` relative to the
  working directory, which is why compose sets the absolute path.

### Pattern A — pinned image tag (use when any code changed)

The candidate is merged to master (or built from its branch and pushed), so
`.github/workflows/docker.yml` has published
`ghcr.io/korjavin/vsbot:sha-<full-commit-sha>`. **This is the default pattern.**
Use it whenever the candidate is anything other than a drop-in net swap.

`docker-compose.canary.yml`:

```yaml
# Canary overlay — brought up ALONGSIDE the incumbent, never instead of it.
# One candidate at a time. Delete this file when the canary is torn down.
services:
  vsbot-canary:
    # An immutable sha- tag, never :latest and never a local `build:`. What the
    # owner judged must be re-runnable byte for byte a month later.
    image: ghcr.io/korjavin/vsbot:${CANARY_TAG:?set CANARY_TAG to the candidate sha- tag}
    container_name: vsbot-canary
    restart: unless-stopped

    environment:
      BACKEND_URL: wss://vs.wandergeek.org/ws
      # The lobby identity. The server appends its own suffix, so this shows up
      # as e.g. "Canary-gen6 Bot 2131". Keep it [A-Za-z0-9-]: it is percent-
      # encoded into the connect URL, but the owner has to recognise it at a
      # glance next to "SuperiorBot Bot 2127".
      BOT_NAME_PREFIX: Canary-gen6

      SEARCH: MCTS
      MCTS_ARTIFACT: /opt/vsbot/artifacts/mcts_champion.json

      # Match the incumbent unless the budget IS the candidate (see §6).
      VSBOT_TURN_MILLIS: "10000"
      VSBOT_PONDER: "false"

      # Accept-only, exactly like production. A canary that challenges would
      # pester real humans with an unjudged bot.
      CHALLENGER: "false"

    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 512M
        reservations:
          memory: 128M

    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

    security_opt:
      - no-new-privileges:true
```

```bash
CANARY_TAG=sha-9602b2fdeadbeef... \
  docker compose -f docker-compose.yml -f docker-compose.canary.yml up -d vsbot-canary
```

The trailing service name is load-bearing: without it compose would also
recreate the incumbent.

### Pattern B — incumbent image, candidate artifact mounted (net-only candidates)

Valid **only** when the candidate differs from the incumbent by the net file
alone. Same image tag as production, plus a read-only bind mount and an
`MCTS_ARTIFACT` override. This skips the image build entirely, which is the
whole point: a nightly generation can be in front of the owner in a minute.

Same overlay as Pattern A, with these three keys changed:

```yaml
services:
  vsbot-canary:
    # SAME tag the incumbent runs — read it off `docker inspect` (§2), do not
    # write :latest here, or the canary silently becomes a code+net A/B.
    image: ghcr.io/korjavin/vsbot:${VSBOT_TAG:?pin the incumbent tag}
    environment:
      MCTS_ARTIFACT: /opt/vsbot/candidate/candidate.json
    volumes:
      # Absolute host path, or ./-relative to this file. If the source file does
      # not exist, Docker creates a *directory* there and startup fails.
      - ./artifacts/champions/gen-6.json:/opt/vsbot/candidate/candidate.json:ro
```

```bash
chmod a+r artifacts/champions/gen-6.json   # the container runs as the unprivileged `vsbot` user
VSBOT_TAG=sha-<incumbent-sha> \
  docker compose -f docker-compose.yml -f docker-compose.canary.yml up -d vsbot-canary
```

A missing or malformed artifact is a **hard startup failure**, not a silent
fallback — the container dies in `docker compose logs` instead of playing at
random under the champion's name. That is by design; do not "fix" it by making
the path optional.

### Host budget

The deploy host is 4 vCPU / 8 GB and runs other stacks. Two bot services at
`cpus: "2.0"` can claim the whole box if both search at once. Keep the canary's
limits identical to the incumbent's anyway — a canary throttled below production
is not the thing the owner is judging — and instead:

- run canaries outside the nightly trainer window (03:00 onward, ~5-7 h,
  `~/Project/nnue-trainer/docs/server-training.md`);
- watch `docker stats --no-stream vsbot vsbot-canary` during the first game;
- treat `VSBOT_PONDER=true` as a doubling of the canary's CPU appetite, because
  it searches during the opponent's turn too.

### Confirm what is actually running

The banner is printed before the first connection precisely so an operator can
*check* the engine and artifact instead of inferring them from move quality:

```bash
docker compose -f docker-compose.yml -f docker-compose.canary.yml logs -f vsbot-canary
```

```
vsbot 0.1.0 starting: url=wss://vs.wandergeek.org/ws search=MCTS challenger=false
vsbot: budget=10000ms/turn split 50/30/20 across up to 3 actions (early-stop=true, extension=true) ponder=off
vsbot: engine=MCTS artifact=/opt/vsbot/candidate/candidate.json …
[vsbot] connected to wss://vs.wandergeek.org/ws?bot=true&namePrefix=Canary-gen6
[Canary-gen6 Bot 2131] registered as Canary-gen6 Bot 2131 (…)
```

Three things to verify before telling the owner it is ready:

1. the `artifact=` path is the **candidate**, not the champion;
2. the lobby name is the canary name;
3. after the first game, both counters are clean:

```bash
docker compose logs vsbot-canary | grep -i 'illegal'          # must be empty
docker compose logs vsbot-canary | grep -i 'fallback_actions' # must be 0
```

A non-zero `fallback_actions` means an action blew its deadline and answered
with the pre-selected fallback instead of a searched move. Fix that before the
owner plays: the games would be judging the deadline, not the candidate.

### Tearing it down

```bash
docker compose -f docker-compose.yml -f docker-compose.canary.yml rm -sf vsbot-canary
```

**Never `docker compose -f docker-compose.yml -f docker-compose.canary.yml down`**
— with both files merged that stops the incumbent too. Note also that a plain
`docker compose up -d` on the base file alone will warn about the canary as an
orphan container; that warning is expected and must not be answered with
`--remove-orphans` while the owner is still playing it.

---

## 4. What the owner does

Tell the owner exactly one thing: **the lobby name** (e.g. `Canary-gen6 Bot 2131`)
and what changed in one sentence.

The protocol:

- **N >= 5 completed games** against the canary. Fewer than 5 is not a canary;
  the precedent that defines this runbook was decided in 3
  (nnue-canary.md:9-12).
- Of those, at least **two adversarial probes** against the known live exploit
  vectors the canary doc names: **base-turtling** and **en-prise gifting**
  (nnue-canary.md, "Branch naming"). A candidate that plays beautifully until
  turtled is exactly the failure this runbook exists to catch.
- At least **one control game vs the incumbent** (`SuperiorBot Bot …`) in the
  same session, so "felt stronger" has a same-day reference point rather than a
  memory of last month's bot.
- Judge on **pace as well as strength**. The owner's UX bound is 10-15 s for the
  full 3-action turn (owner directive, 2026-08-13). A canary that is stronger
  and unpleasant to play is a REJECT, and the bound itself is revisable only by
  the owner.

There is no win-rate threshold, and adding one would recreate the Goodhart
failure at human scale. The verdict is a judgement.

---

## 5. Recording the verdict

The verdict goes in **the candidate's bead** (not this standing bead,
`vsbot-eiw`), appended so the history of a candidate that took two canaries
stays readable:

```bash
bd update vsbot-<candidate> --append-notes "CANARY VERDICT 2026-08-20 — PROMOTE
candidate: gen-6 (artifacts/champions/gen-6.json), image sha-9602b2f…
deployed as: Canary-gen6 Bot 2131, Pattern B (net-only, incumbent tag sha-1a2b3c4)
games: 6 (2 turtle probes, 1 gifting probe, 1 control vs SuperiorBot)
owner impressions: punished the turtle line by turn 20 instead of shuffling;
  still gifts an occasional en-prise cell in the opening; pace fine, ~9 s/turn.
gates: A 0.571/400 (gen-6 report), B(i) 0.94/400 vs ab-enhanced, B(ii) no regression
rollback target: VSBOT_TAG=sha-1a2b3c4…"
```

Required fields, every time: **date**, **verdict** (`PROMOTE` | `REJECT` |
`ITERATE`), **impressions in the owner's words**, the games played, and the
rollback target.

`REJECT` and `ITERATE` are recorded with the same weight as `PROMOTE` — a
rejected candidate whose failure mode is written down is the only thing that
stops the next generation from re-learning it. Tear the canary down (§3) and
leave the incumbent exactly as it was; no other action is needed, because the
canary never touched the default.

---

## 6. On PROMOTE

Four steps: history, default, image, pin. Do them in that order — the artifact
must be in git before the image that bakes it is built.

### 6a. Artifact history and the default

**Convention (becomes real at the first promotion):**

| Path | Role |
|---|---|
| `artifacts/champions/gen-<N>.json` | Immutable history. One file per promoted generation, never edited, never deleted. |
| `artifacts/mcts_champion.json` | The default the bot loads. A **byte-identical copy** of the newest promoted `gen-<N>.json`. |

A copy, not a symlink: `Dockerfile` does `COPY artifacts /opt/vsbot/artifacts`,
and a promotion mechanism whose correctness depends on how a builder treats
symlinks inside a copied directory is not a mechanism worth debugging at 1 a.m.
The 700 KB duplicate is the price.

```bash
cd /home/devbox/Project/vsbot
mkdir -p artifacts/champions
cp -f /path/to/candidate.json artifacts/champions/gen-6.json
cp -f artifacts/champions/gen-6.json artifacts/mcts_champion.json
sha256sum artifacts/champions/gen-6.json artifacts/mcts_champion.json  # must match
```

For an RL candidate the source is the trainer volume, exactly as the predecessor
runbook does it (`~/Project/nnue-trainer/docs/server-training.md`, "Promoting an
RL champion to the live bot"):

```bash
docker cp nnue-trainer-trainer:/work/out/champion_gen6.json artifacts/champions/gen-6.json
```

**Also at first use:** add `artifacts/champions` to `.dockerignore`. The whole
`artifacts/` directory is baked into the image today (it is explicitly *not*
ignored), so without that line every promotion permanently grows the image by
another champion. The history belongs in git; only the default belongs in the
image.

Commit with the provenance in the message — the bead and the commit are the two
places a future reader will look:

```bash
git add artifacts/champions/gen-6.json artifacts/mcts_champion.json .dockerignore
git commit -m "promote gen-6 champion (owner canary 2026-08-20)

Gate A 0.571/400 vs gen-5. Gate B: 0.94/400 vs ab-enhanced, no regression vs
the Go bot. Owner verdict PROMOTE, recorded in bd vsbot-<candidate>.
Rollback: VSBOT_TAG=sha-1a2b3c4…"
```

### 6b. Image and pin

Merging to master publishes both tags from `.github/workflows/docker.yml`:
`ghcr.io/korjavin/vsbot:latest` and the immutable
`ghcr.io/korjavin/vsbot:sha-<full-commit-sha>`. Wait for that workflow to go
green, then pin the deployment to the **immutable** tag rather than riding
`:latest` — a pinned deployment is one that can be reasoned about after the next
unrelated merge:

```bash
cd /home/devbox/Project/vsbot
# .env is gitignored; edit the key in place rather than appending a duplicate.
grep -q '^VSBOT_TAG=' .env \
  && sed -i 's|^VSBOT_TAG=.*|VSBOT_TAG=sha-<promotion-commit-sha>|' .env \
  || echo 'VSBOT_TAG=sha-<promotion-commit-sha>' >> .env

docker compose pull && docker compose up -d
docker compose logs -f vsbot     # confirm the banner names the new artifact
```

Then tear the canary down (§3) and record the promotion — new default tag, new
rollback target — back in the bead.

### 6c. Promoting a behaviour change instead of a net

Nothing to copy. The change is a default in `docker-compose.yml` (which is
where the production values live), so promotion is an ordinary PR to that file
plus a redeploy:

```bash
# e.g. promoting pondering after a PROMOTE verdict:
#   VSBOT_PONDER: ${VSBOT_PONDER:-false}   ->   ${VSBOT_PONDER:-true}
docker compose up -d
```

Confirm from the banner, which spells out the canary status of the flag:
`ponder=on (CANARY: search runs during the opponent's turn)`.

---

## 7. Rollback

Rollback has two levels. Do the first immediately; the second only if the
regression is confirmed.

**Level 1 — config, seconds, no build.** Re-pin to the tag recorded in §2. The
container is stateless (no volumes, no ports, all config in env), so there is
nothing to migrate back:

```bash
cd /home/devbox/Project/vsbot
sed -i 's|^VSBOT_TAG=.*|VSBOT_TAG=sha-<pre-promotion-sha>|' .env
docker compose pull && docker compose up -d
docker compose logs -f vsbot     # banner must name the OLD artifact
```

Candidate tags: `gh api /users/korjavin/packages/container/vsbot/versions`, or
the GHCR package page.

**Level 2 — durable, only after Level 1 has stopped the bleeding.** While the
deployment is pinned, `:latest` still carries the bad default, so the next
person who deploys unpinned re-ships it. Revert the promotion commit on master:

```bash
git revert <promotion-commit-sha>     # restores the previous mcts_champion.json
```

`artifacts/champions/gen-<N>.json` stays in git — the history is immutable, and
a rolled-back generation is exactly the kind of thing the next candidate's bead
needs to cite. Record the rollback and its reason in the candidate's bead as a
second `--append-notes` entry, with the same fields as the verdict.

---

## 8. Behaviour changes canary exactly like nets

A longer-thinking or pondering bot is a behaviour change only the owner
promotes (`superiority.md` §4 Gate C: "New live budgets are canaried the same
way as new nets"). Same overlay, same lobby name discipline, same verdict
recording. Only two things differ:

- **The image tag is the incumbent's.** The candidate is the environment, so
  keeping the code identical is what makes the A/B mean anything.
- **Gate A/B do not apply as written.** They compare artifacts; a budget change
  is measured by the S2-T2 sims->Elo cells (`superiority.md` S2) and by the
  arena's fixed-time mode. Record whatever measurement exists, then canary.

The overlay for the documented "Owner canary" budget profile (`DEPLOY.md`,
"Budget profiles"):

```yaml
services:
  vsbot-canary:
    image: ghcr.io/korjavin/vsbot:${VSBOT_TAG:?pin the incumbent tag}
    container_name: vsbot-canary
    environment:
      BACKEND_URL: wss://vs.wandergeek.org/ws
      BOT_NAME_PREFIX: Canary-ponder
      SEARCH: MCTS
      MCTS_ARTIFACT: /opt/vsbot/artifacts/mcts_champion.json
      VSBOT_TURN_MILLIS: "15000"
      VSBOT_PONDER: "true"
      VSBOT_PONDER_SECS: "30"
      CHALLENGER: "false"
```

Candidates that belong in this lane, not in a merge: `VSBOT_TURN_MILLIS`
changes, `VSBOT_PONDER`, `VSBOT_EARLY_STOP` / `VSBOT_EXTENSION` flips, and any
switch to a per-action override (`VSBOT_MOVE_MILLIS`), which disables the
intra-turn allocator and visibly changes the bot's pace.

---

## Conventions this document establishes

Things that do not exist in the repo yet. Each becomes real at its first use;
until then it is a promise this runbook is making on the repo's behalf.

| Convention | Becomes real when |
|---|---|
| `artifacts/champions/gen-<N>.json` history directory | the first canary PROMOTE |
| `artifacts/mcts_champion.json` is a byte-identical copy of the newest `gen-<N>.json` | the first canary PROMOTE |
| `artifacts/champions` listed in `.dockerignore` | the first canary PROMOTE |
| `docker-compose.canary.yml` (untracked, one candidate at a time, deleted at teardown) | the first canary |
| `CANARY_TAG` as the overlay-only tag variable (`VSBOT_TAG` stays the production pin) | the first Pattern A canary |
| `BOT_NAME_PREFIX=Canary-<candidate>` as the lobby identity | the first canary |
| Verdict block appended to the candidate's bead in the §5 format | the first canary |

Everything else this runbook names — `VSBOT_TAG`, `MCTS_ARTIFACT`,
`BOT_NAME_PREFIX`, `SEARCH`, `VSBOT_PONDER`, `VSBOT_TURN_MILLIS`, the
`sha-<commit>` tags, `/opt/vsbot/artifacts`, the hard-fail-on-bad-artifact
behaviour — exists today and is documented in `DEPLOY.md`.
