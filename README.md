# CMR phantom water-level analysis

A short experiment: the water level in a clear cylindrical column rises and
falls under a pulsatile flow driver, with a steel ruler dipped in for scale.
The water carries a faint red dye to make the meniscus easier to see.

The repo contains a small toolchain to extract water level vs time from the
video and convert it to a volume change using the column's inner
cross-section.

## Workflow

```sh
# 1. drop reshot.MOV next to this README (not committed; see .gitignore)
# 2. annotate the meniscus on ~130 keyframes (every 0.5 s)
python3 results/annotate.py

# 3. interpolate + smooth + render outputs
python3 results/process_clicks.py
```

`annotate.py` autosaves on every click, so you can stop and resume. Keys:
**click** to record, **z** to undo, **s** to skip an occluded frame,
**a / d** (or arrow keys) to step prev/next, **q** to save and exit.

## Files

| Path | Description |
| --- | --- |
| `results/annotate.py` | OpenCV mouse-click meniscus annotation tool |
| `results/process_clicks.py` | Loads `clicks.json`, interpolates per-frame, renders outputs |
| `results/clicks.json` | The clicks (frame, time, pixel-y) |
| `results/level_and_volume.png` | Main plot: level + ΔV vs time |
| `results/level_volume.csv` | Per-frame data |

Source video (`reshot.MOV`) and the rendered overlay (`annotated.mp4`) are
kept out of the repo via `.gitignore` to keep clones small. Drop the video
next to this README and re-run the two scripts to regenerate everything,
including `annotated.mp4` (a 1152×648 mp4v overlay showing the meniscus
line, level, ΔV, and a moving time marker).

## Method

1. **Calibration** — nine cm-major tick rows on the ruler were identified by
   hand from a still frame and fit with a 2nd-degree polynomial
   `cm = f(pixel_y)` (the ruler tilts slightly in perspective, so the
   pixels-per-cm changes from ~125 at the top to ~75 at the bottom).
2. **Annotation** — every 0.5 s, the user clicks the meniscus on a high-zoom
   crop of the ruler region. Around 130 clicks for a 64 s clip.
3. **Interpolation** — per-frame level by linear interpolation between
   clicks, then a 25-point Savitzky-Golay (order 3) low-pass.
4. **Volume** — cylindrical column, inner diameter `D = 9.0 cm` (measured),
   so `A = π (D/2)² ≈ 63.62 cm² = 63.62 mL/cm`. Then
   `ΔV(t) = A · (h(t) − h₀)`, where `h₀` is the median of the first 2 s.

## Results

| Quantity | Value |
| --- | --- |
| Water level range | 14.18 – 17.07 cm (swing ≈ 2.9 cm) |
| ΔV range | −19 – +165 mL (swing ≈ 184 mL) |
| Period | ≈ 7.7 s (≈ 7.8 cycles/min) |
| Baseline (h₀) | 14.62 cm |

## Why manual?

I started with several auto-detection approaches (V-step, redness-step,
V·R conjunction, specular-shine peak). Each could reproduce the broad
periodic structure but kept being fooled at peaks by competing features —
the wet-film boundary above the actual meniscus, the cup rim, glare on
the ruler, and the dye-density gradient near the cup bottom. A search
window seeded from prior detections cured one mode and introduced another.
Manual annotation took about 10 minutes of clicking and produced a
visibly clean curve, with cycle-to-cycle consistency well below the
±0.5 cm calibration error.

The auto-detector is recoverable from git history if you want to use it
as a starting point for a different video.
