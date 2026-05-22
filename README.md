# CMR phantom water-level analysis

A short experiment: track the rising and falling water level in a clear cup
with a steel ruler dipped in it, while a displacer (orange ball) is moved
in and out of the water. The water has a faint red dye to make the wet/dry
boundary on the ruler easier to see.

The analysis script extracts water level vs time from the video, then
converts it to a volume change using the cup's inner cross-section.

## Files

| Path | Description |
| --- | --- |
| `reshot.MOV` | Source video (1920x1080, 30 fps, 64 s) |
| `results/analyze.py` | Full pipeline (calibration, detection, plotting) |
| `results/level_and_volume.png` | Main plot: level, ΔV, and detector confidence vs time |
| `results/level_volume.csv` | Per-frame data |
| `results/overlay_montage.jpg` | Six sample frames with the detected meniscus drawn |

## Method

1. **Calibration** — nine cm-major tick rows on the ruler were identified by
   hand from a still frame and fit with a 2nd-degree polynomial
   `cm = f(pixel_y)` (the ruler tilts slightly in perspective, so the
   pixels-per-cm changes from ~125 at the top to ~75 at the bottom).
2. **Meniscus detection** — every 3rd frame (~10 Hz), inside a fixed ROI on
   the ruler shaft, find the row with the strongest bright-to-dark step in
   brightness (averaged across the ROI width), plus a redness-step term.
   A search window seeded from the previous detection (±180 px) prevents
   spurious far-away matches.
3. **Smoothing** — 5-pt median filter, then 11-pt Savitzky-Golay (order 2).
4. **Volume** — assume a cylindrical cup with inner diameter `D = 8 cm`, so
   `A = π (D/2)² ≈ 50.27 cm² = 50.27 mL/cm`. Then `ΔV(t) = A · (h(t) − h₀)`,
   where `h₀` is the median level over the first 2 s of the analysis window.

## Results

| Quantity | Value |
| --- | --- |
| Analysis window | t = 10 s … 64 s (dye-mixing settling at start excluded) |
| Water level range | 14.0 – 18.6 cm (swing 4.6 cm) |
| ΔV range | −132 – +99 mL (swing ≈ 231 mL) |
| Period | ≈ 7.8 s (≈ 7.7 cycles/min) |
| Baseline (h₀) | 16.67 cm |

The first cycle (~t = 5 s) reads ~1 cm high because dye splashes during
mixing produced a misleading brightness step; that's why the first 10 s
are excluded from the summary stats.

## Reproducing

```sh
cd results
python3 analyze.py
```

Dependencies: Python 3 with `opencv-python`, `numpy`, `scipy`, `matplotlib`.

## Caveats

- Cup inner diameter (8 cm) was supplied by hand. A direct measurement would
  improve the volume calibration.
- Detector confidence dips during troughs because less of the ruler is
  wet/red; the smoothed trace stays reasonable but raw values are noisier.
- The 2nd-degree perspective fit has residuals up to ±0.4 cm in places —
  the absolute level numbers should be read as approximate (±0.5 cm).
  The swing and period are robust.
