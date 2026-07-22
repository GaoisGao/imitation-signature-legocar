# Smoothing a Signature Path for a Nonholonomic Pen-Car

Reference notes for preprocessing a raw recorded signature trajectory before
using it as the reference path / training the imitation policy. These are
suggestions only — no code has been changed to match them yet.

---

## 1. Why raw corners are a problem

The camera path is (a) noisy at the pixel / sub-mm level and (b) full of
near-instant direction reversals that a human wrist can do but a
differential-drive car **cannot**:

- The car can't translate sideways; heading only changes by turning, at a
  bounded rate.
- At forward speed `v`, the achievable path curvature is limited:
  `κ_max ≈ ω_max / v` (turn-rate limit). A hairpin implies `κ → ∞`, which is
  infeasible.
- The pen sits at an **offset** from the chassis, so any sharp / in-place
  heading change smears the tip through an arc (and can drag / skip the pen).

If the reference is infeasible, the pure-pursuit "expert" will cut / overshoot
those corners inconsistently → the behavior-cloning target becomes noisy and
contradictory → the policy learns a blurred, drifting version. **Smoothing is
really "projecting the human path onto the set of paths the car can actually
drive."**

---

## 2. Recommended filter stack (in order)

Apply as a pipeline; each stage assumes the previous one ran:

| Step | Purpose | Suggested method |
|------|---------|------------------|
| 0. Outlier rejection | Kill single-frame camera mis-detections (spikes) | **Median filter** (window 3–5 samples) on x,y — do this *before* averaging, since a mean smears a spike instead of removing it |
| 1. Metric resampling | Uniform spacing so smoothing is isotropic | **Arc-length resample** (already in the pipeline) to ~1–2 mm spacing |
| 2. Denoise / smooth | Remove jitter, keep shape | **Savitzky–Golay** *or* a **smoothing spline** (see §3) |
| 3. Corner feasibility | Round the sharp turns to respect `κ_max` | **Curvature limiting / clothoid (Euler-spiral) fillets**, or locally stronger smoothing where `κ > κ_max` |
| 4. Derived features | Give the controller / policy tangent + curvature | Compute analytically from the spline (not finite differences on raw points) |
| 5. Speed profile | Slow down in tight regions | `v(s) = min(v_max, sqrt(a_lat_max / κ(s)))` |

---

## 3. Which smoothing filter to use

**First choice: a smoothing cubic B-spline** (`scipy.interpolate.splprep(..., s=S)`).
- Fits a C²-continuous curve; you get **analytic tangent (heading) and
  curvature** for free — exactly what pure pursuit and the observation vector
  need.
- One intuitive knob `s` (total allowed squared deviation). Scale it with point
  count and noise, e.g. `s ≈ N * (σ_mm)²` where `σ_mm` is camera noise (~0.5–1 mm).
- Naturally rounds corners; increase `s` to round more.

**Strong second: Savitzky–Golay** (`scipy.signal.savgol_filter`, polyorder 2–3,
odd window).
- Preserves peaks / extrema far better than a moving average (which shrinks
  loops and flattens the signature's character).
- Can output smoothed derivatives directly (`deriv=1`) for heading.
- Cheaper / simpler than splines; good for staying close to an array-based flow.

**Avoid relying on the plain moving average alone.** It's a crude low-pass: it
rounds *everything* uniformly, biases corners inward, and its derivatives are
noisy. Fine as a first pass, not ideal as the final smoother for a
curvature-sensitive controller.

---

## 4. Handling the *sharp corners* specifically

Global smoothing strong enough to fix a hairpin will over-smooth the gentle
parts. Make corner-rounding **adaptive**:

1. Compute curvature `κ(s)` along the path.
2. Set a feasibility bound `κ_max = ω_max / v_nominal` (derive `ω_max` / `v`
   from motor limits + wheel radius / track width in the MuJoCo XML).
3. Wherever `κ > κ_max`, replace that corner with a **clothoid (Euler spiral)
   or circular-arc fillet** of radius `1/κ_max`, or apply a locally larger
   smoothing window. Clothoids are the "nice" option because curvature ramps
   linearly (no curvature jump), which is easy for the car to follow.

Pragmatic cheaper substitute: **iterated corner-cutting** (Chaikin's algorithm,
2–3 iterations) — rounds sharp vertices while leaving straight runs mostly
intact, then feed the result to the spline.

---

## 5. Pitfalls to watch

- **Smooth in millimeters, not pixels.** Do it after the homography (the
  pipeline already records paper-mm — good).
- **Don't over-smooth.** Too much and the signature stops being recognizable /
  loses loops. Tune visually against the raw overlay (the auto-saved PNG in
  `datasets/plots/` is perfect for this).
- **Keep demo and rollout preprocessing identical.** Whatever filter is chosen
  must be applied the same way to the reference used at training *and* at
  deployment, or the policy sees a distribution shift.
- **Treat pen-up gaps separately** if added later — smooth each stroke
  independently so you don't interpolate across a lift.
- **Pure pursuit already rounds corners** via its lookahead, so heavy
  pre-smoothing can be partly redundant. Think of pre-smoothing as guaranteeing
  *feasibility / consistency*, and lookahead as the runtime tracker; tune them
  together (bigger lookahead ↔ less aggressive pre-smoothing).

---

## 6. What should actually enter the policy

Feed the **post-smoothing, arc-length-parameterized** path plus derived
quantities, not raw points:

```
s (arc length), x_mm, y_mm, tangent_heading θ(s), curvature κ(s), desired_speed v(s)
```

This matches the pipeline doc's "resample by arc length" target format and gives
the observation vector clean, feasible geometry to condition on.

---

## 7. Suggested concrete recipe to try first

```
median(5)
  → arc-length resample @ 1.5 mm
  → smoothing B-spline (s ≈ N · 0.5²)
  → curvature-limit corners to 1/κ_max via clothoid / arc fillets
  → derive θ, κ
  → speed profile with a_lat_max
```

Validate each candidate by:
- overlaying smoothed vs. raw on the PNG, and
- running pure pursuit on it in simulation before generating BC data.
