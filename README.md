# op-model-grader

Grade your openpilot model's driving **against your own manual driving**, from
the same rlogs. You are the ground truth: the tool finds comparable maneuvers
in both the engaged and manual parts of your drives, computes driving-quality
metrics for each side, and scores the model relative to you. Output is a
single self-contained HTML report.

This is a personal side project, vibe-coded with Claude Code. Free to use,
fork, whatever — at your own risk, no warranty, not affiliated with comma.ai.
It reads logs; it never touches your car.

## Why rlogs (not qlogs)

rlogs carry every message at full rate — `carState` at 100 Hz — which is what
you need to measure jerk, stop lurch, and steering oscillation. qlogs are
decimated too heavily for any of that.

## Install

Easiest — grab the code, then run the launcher for your OS (it sets everything
up on first run and opens the app):

```
git clone https://github.com/SpysyWeeb/op-model-grader && cd op-model-grader
./start.sh        # Linux / macOS
start.bat         # Windows (or double-click it in Explorer)
```

Run a launcher with arguments to use the CLI instead of the UI, e.g.
`./start.sh testdata/demo -o report.html`.

Manual install, if you prefer:

```
git clone https://github.com/SpysyWeeb/op-model-grader && cd op-model-grader
python3 -m venv .venv && . .venv/bin/activate   # or: uv venv && . .venv/bin/activate
pip install -e .
```

No openpilot checkout needed: the cereal/opendbc capnp schemas are vendored
and decoded with standalone pycapnp. Works with `.bz2` (pre-2024) and `.zst`
rlogs; unknown/fork-custom message types are skipped, and fields that don't
exist in a given openpilot era degrade to "insufficient data" instead of
crashing.

## Usage

```
# local files, dirs (scanned recursively), or globs; multiple routes aggregate
opgrader /path/to/rlogs/ another/route/dir -o report.html --open

# straight from comma connect (JWT from ~/.comma/auth.json, or pass --jwt)
opgrader --route "a2a0ccea32023010|2023-07-27--13-01-19" -o report.html

# or skip the CLI entirely:
opgrader --ui
```

### Simple UI

`opgrader --ui` opens a small desktop window (plain Python/Tkinter — nothing
extra to install beyond your distro's `python3-tk` package, which most
systems already have). The whole flow is point-and-click:

1. **Sign in once** — if `~/.comma/auth.json` doesn't have a token yet, the
   window shows a paste box and a button that opens https://jwt.comma.ai;
   what you paste is validated and saved (file created with 0600
   permissions, other keys in that JSON left alone).
2. **Browse drives** — pick a device, see your recent drives newest-first
   with start time, duration, segment count, git branch, and an rlog
   availability badge ("rlogs ready 12/12", "rlogs 4/12", "qlogs only").
3. **Request uploads** — select drives whose rlogs are still on the device
   and click "Request upload". Uploads happen when the device is on WiFi
   unless you tick "upload over cell data". If the device is offline you'll
   be told to start the car and try again; the badge refreshes itself every
   ~30 s while a request is pending.
4. **Grade** — select one or more drives with rlogs ready (partial ≥80% is
   allowed, with a warning), optionally add local rlog folders, hit Grade.
   Progress (download → decode → grade) is shown live; the finished report
   opens in your browser and stays in the "past reports" list
   (`~/.cache/opgrader/reports/`, double-click to reopen).

Everything runs on your computer; your JWT is only ever sent to
api.comma.ai and athena.comma.ai, never anywhere else.

## How grading works

For each metric the model's aggregate `m` (median across its events) is
compared to yours `d`:

```
r = m / max(d, eps)              # lower-is-better; inverted if higher is better
                                 # "match" metrics (follow gap, peak decel + its
                                 # timing, launch time-to-speed, unwind rate) use
                                 # r = max(m,d)/min(m,d): deviating from the
                                 # driver in either direction is penalized
score = 100                      if r <= 1        (as good as you or better)
score = 100 - 50*(r-1)           if 1 < r <= 2    (twice your number = 50)
score = max(0, 50 - 25*(r-2))    if r > 2
```

Letters: ≥93 A, ≥85 A−, ≥78 B+, ≥70 B, ≥60 C, ≥50 D, else F. A ratio metric
needs at least 3 events on each side, otherwise it's shown greyed-out as
insufficient data. Metrics with no human counterpart (driver-rescue rate,
missed turn-ins) or where your baseline is ~zero (S-curve overshoot) use
documented absolute scales instead.

Two top-level grades, each from weighted categories (empty ones are dropped
and weights renormalized; overall = ½ longitudinal + ½ lateral):

**Longitudinal**
- **Smoothness** (0.30) — RMS/P95 jerk, accel reversals/min, % time |a|>2
- **Following** (0.20) — median time gap, gap hunting, accel reversals in
  follow, and **follow-distance adherence**: the model's held gap (inverted
  from the long-MPC distance formula over steady-follow samples) vs the
  ACTIVE personality's target, scored absolutely (100 at ≤5% error, 50 at
  25%, 0 at ≥50%). Targets are **fork-dependent** — stock defaults are
  aggressive 1.25 / standard 1.45 / relaxed 1.75 s; set yours with
  `--t-follow aggressive=1.0,standard=1.45,relaxed=2.0`, the UI's target
  boxes, or `~/.config/opgrader/config.json`.
- **Stopping** (0.20) — peak decel, decel timing, stop lurch (max |jerk| in the
  last 2 s before standstill), accel at crawl speed
- **Launch** (0.17) — time to 5 m/s, peak jerk
- **Responsiveness** (0.13) — lead-decel response latency, pull-away latency

**Lateral**
- **Ping-Pong** (0.40) — steering-wheel oscillation (angle minus its centered
  2 s moving average) per speed bin, engaged vs manual: oscillation RMS and
  >3° swing reversals/min. Scored per bin, time-weighted; the report calls out
  your worst speed range and breaks 0–10 mph down to 1 mph resolution.
- **Turn Execution** (0.30) — for sharp turns (peak ≥90° starting below
  15 mph): S-curve overshoot after the unwind, recovery wobbles, unwind rate,
  and the driver-rescue rate (how often a human had to grab the wheel during
  the straighten-out). 20–90° curve episodes are reported separately.
- **Turn-In Timing** (0.20) — blinker below 20 mph opens an intent window,
  classified by integrated yaw into intersection turn / lane change /
  ambiguous. Measures turn-in delay vs your own habit and the missed-turn-in
  rate (driver had to start the turn).
- **General Smoothness** (0.10) — RMS lateral jerk (vEgo × yaw rate),
  steering-rate RMS and steering reversals above 10 m/s, % time |lat accel|>3.

Turn definitions match my on-device analyzer so numbers are comparable:
onset at |angle| ≥ 20° (actual or commanded, whichever first), unwind point at
50% of that signal's peak, sharp turn = peak ≥ 90° below 15 mph, positive
angle = left (ISO). On torque-controlled cars the commanded angle is
reconstructed from `carControl.actuators.curvature` with a minimal port of
opendbc's `VehicleModel` (MIT) built from your logged `carParams`.

### Always-On-Lateral / MADS

Attribution is per axis: longitudinal metrics follow `carControl.longActive`,
lateral metrics follow `carControl.latActive`. So AOL/MADS time counts as the
model for steering and as *you* for gas/brake — exactly what you want. Events
with mixed control over their window are discarded. Old logs without the
per-axis flags fall back to the single `enabled` flag (noted in the report).

### Mode & personality breakdowns

Experimental/chill mode and the longitudinal personality are tracked **per
sample** — mid-drive hot-swaps are expected and handled. Every longitudinal
event is tagged with the mode/personality active during its window (≥90%
agreement, else "mixed"); mixed events stay in the overall grades but are
excluded from the buckets. The report adds a "Longitudinal breakdowns"
section grading each mode and personality bucket against the same human
baseline (per-bucket n≥3 gating), plus time-in-mode/personality chips in the
header. Old logs that predate the personality field are reported as
personality-unknown rather than guessed.

### Driving-model identification

The report header names the driving model when it can, in layers:

1. **Fork model selectors** — sunnypilot's model manager and FrogPilot
   persist the selected model in the device params, which land in the log's
   initData; only those whitelisted selector keys are ever read (the params
   dump contains secrets and is never embedded in reports).
2. **Build commit** — stock/non-switcher forks commit models as git-lfs
   pointers, so the pointer's sha256 at `initData.gitCommit` is fetched from
   GitHub (cached in `~/.cache/opgrader/modelid.json`, works for
   force-push-orphaned commits) and reverse-looked-up in the bundled
   `opgrader/model_hashes.json`.
3. Otherwise the sha256/commit is shown so a human can identify it — and on
   switcher forks without a persisted selection, logs simply can't tell.

Everything is best-effort: offline grading is never blocked. Dirty builds
get a "hash may not match device" caveat. To teach it new models, add
`{"<sha256>": {"name": ..., "source": ...}}` entries to `model_hashes.json`
(only with verified provenance).

## Limitations

- Needs **both** engaged and manual driving in the logs; whatever side is
  missing shows up as "insufficient data" rather than a made-up grade.
- If `openpilotLongitudinalControl` is false, "engaged" gas/brake is your
  car's stock ACC, not openpilot — the report puts a banner up and grades
  anyway.
- Tolerates schema drift across openpilot versions (2023 releases through
  current master tested), but truly ancient logs may lack whole channels.
- If you drive with AOL on all the time there is no manual steering baseline
  at speed; ratio-scored lateral metrics will be data-starved (absolute ones
  like rescue rate still work).
- The human baseline is *your* driving, bad habits included. A 100 doesn't
  mean perfect; it means "at least as smooth as you".

## Credits

- Charts by [uPlot](https://github.com/leeoniya/uPlot) (MIT), vendored in
  `opgrader/assets/uplot/`.
- capnp schemas vendored from [commaai/cereal](https://github.com/commaai/openpilot)
  and [commaai/opendbc](https://github.com/commaai/opendbc) (MIT).
- `opgrader/vehicle_model.py` is a minimal port of opendbc's vehicle model (MIT).

MIT license. Drive safe; this tool only watches.
