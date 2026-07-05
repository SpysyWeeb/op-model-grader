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

### Web UI

`opgrader --ui` (optionally `--port 8385`) starts a small local server and
opens your browser. From there the whole flow is point-and-click:

1. **Sign in once** — if `~/.comma/auth.json` doesn't have a token yet, the
   page walks you through getting one (https://jwt.comma.ai) and saves what
   you paste (file is created with 0600 permissions; other keys in that JSON
   are left alone).
2. **Browse routes** — pick a device, see your recent drives newest-first
   with start time, duration, segment count, git branch, and an rlog
   availability badge ("rlogs ready 12/12", "rlogs 4/12", "qlogs only").
3. **Request uploads** — for drives whose rlogs are still on the device,
   one click asks the device (via athena) to upload them. Uploads happen
   when the device is on WiFi unless you tick "upload over cell data". If
   the device is offline you'll be told to start the car and try again; the
   badge refreshes automatically (~30 s) while the request is pending.
4. **Grade** — tick one or more routes with rlogs ready (partial ≥80% is
   allowed, with a warning), optionally add local rlog directories, hit
   Grade. Progress (download → decode → grade) is shown live; the finished
   report opens in a new tab and stays in the "past reports" list
   (`~/.cache/opgrader/reports/`).

Security: the server binds to **127.0.0.1 only** — nothing on your network
can reach it — and your JWT is only ever sent to api.comma.ai and
athena.comma.ai, never anywhere else.

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
- **Following** (0.20) — median time gap, gap hunting, accel reversals in follow
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
