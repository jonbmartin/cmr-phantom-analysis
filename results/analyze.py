"""Analysis pipeline for the CMR phantom water-level video.

Steps:
  1. Read reshot.MOV
  2. Calibrate cm-from-pixel using a hand-identified set of cm-tick rows
     (quadratic fit, since the camera looks slightly down so the ruler
     foreshortens toward the bottom of the frame).
  3. For every 3rd frame, detect the water meniscus on a fixed ROI on the
     ruler by combining a brightness step (bright dry ruler -> darker wet
     ruler) and a redness step (red-dyed water below the line).  A search
     window seeded from the previous detection prevents far-away spurious
     matches.
  4. Smooth (5-pt median + 11-pt Savitzky-Golay).
  5. Convert level to a volume change with A = pi*(D/2)^2.
  6. Output:
        level_and_volume.png    - main plot
        level_volume.csv        - per-frame data
        overlay_montage.jpg     - six sample frames with the detected line
        annotated.mp4           - the full video with the meniscus line,
                                  level, and delta-V overlaid

Configuration in the CONFIG block at the top.
"""
import cv2, numpy as np, csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

# ---------------------- CONFIG ----------------------
HERE = Path(__file__).resolve().parent              # results/
REPO = HERE.parent                                  # cmr_phantom_analysis/
VIDEO_IN = REPO / 'reshot.MOV'
OUT = HERE
OUT.mkdir(exist_ok=True, parents=True)

D_CM = 9.0                                          # cup inner diameter (cm)
A_CM2 = np.pi * (D_CM/2)**2                         # 63.617 cm^2

DETECT_STEP = 3                                     # detect every Nth frame
T_VALID_START = 10.0                                # seconds; skip dye-mixing

# Ruler ROI in full-frame pixel coords (the cm-scale half of the shaft)
RULER_X0, RULER_X1 = 1095, 1255
Y_TOP, Y_BOT = 40, 1070

# Hand-identified cm-tick rows on a calibration frame at t=2 s.
CAL_CM = np.array([21, 20, 19, 18, 17, 16, 15, 14, 13], dtype=float)
CAL_Y  = np.array([62, 183, 322, 445, 658, 788, 892, 974, 1049], dtype=float)

# Annotated-video output
ANNOTATED_OUT = OUT / 'annotated.mp4'
ANNOTATED_SCALE = 0.6                               # downscale factor
ANNOTATED_FOURCC = 'mp4v'

# ---------------------- CALIBRATION ----------------------
_coeff = np.polyfit(CAL_Y, CAL_CM, 2)
y_to_cm = np.poly1d(_coeff)
print(f'calibration residuals (cm): '
      f'max={float(np.abs(y_to_cm(CAL_Y)-CAL_CM).max()):.3f}')

# ---------------------- DETECTOR ----------------------
def detect_meniscus(fr, prev_y=None, search_margin=180):
    """Find the meniscus row in full-frame coordinates + confidence in [0,1]."""
    roi = fr[Y_TOP:Y_BOT, RULER_X0:RULER_X1].astype(np.float32)
    B, G, R = roi[..., 0], roi[..., 1], roi[..., 2]
    V = np.maximum(np.maximum(B, G), R).mean(axis=1)
    V_s = savgol_filter(V, 21, 2)

    redness = (R - 0.5 * (G + B)).mean(axis=1)
    red_s = savgol_filter(redness, 21, 2)

    OFF = 25
    N = len(V_s)
    above_V = np.array([V_s[max(0, i-OFF):i].mean() if i > 0 else V_s[0]
                        for i in range(N)])
    below_V = np.array([V_s[i+1:min(N, i+OFF+1)].mean() if i+1 < N else V_s[-1]
                        for i in range(N)])
    step_V = above_V - below_V                       # +ve at bright->dark step

    above_R = np.array([red_s[max(0, i-OFF):i].mean() if i > 0 else red_s[0]
                        for i in range(N)])
    below_R = np.array([red_s[i+1:min(N, i+OFF+1)].mean() if i+1 < N else red_s[-1]
                        for i in range(N)])
    step_R = below_R - above_R                       # +ve when below is redder

    s_V = step_V / max(1.0, step_V.max())
    s_R = step_R / max(1.0, step_R.max())
    score = 0.65 * np.clip(s_V, 0, None) + 0.35 * np.clip(s_R, 0, None)

    if prev_y is not None:
        lo = max(0, (prev_y - Y_TOP) - search_margin)
        hi = min(len(score), (prev_y - Y_TOP) + search_margin)
        mask = np.zeros_like(score, dtype=bool); mask[lo:hi] = True
        score = np.where(mask, score, -1.0)

    best = int(np.argmax(score))
    return Y_TOP + best, float(score[best])

# ---------------------- RUN DETECTION ----------------------
cap = cv2.VideoCapture(str(VIDEO_IN))
fps = cap.get(cv2.CAP_PROP_FPS)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'video: {n_total} frames @ {fps:.2f} fps, {W}x{H}')

times, ys, confs = [], [], []
prev_y = None

overlay_targets = set()
for t_target in [0.5, 5.0, 15.0, 30.0, 45.0, 60.0]:
    f_target = (int(t_target * fps) // DETECT_STEP) * DETECT_STEP
    overlay_targets.add(f_target)
overlays = {}

idx = 0
while idx < n_total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        idx += DETECT_STEP; continue
    y_abs, conf = detect_meniscus(fr, prev_y=prev_y)
    if conf < 0.05 and prev_y is not None:
        y_abs = prev_y; conf = 0.0
    else:
        prev_y = y_abs

    times.append(idx / fps); ys.append(y_abs); confs.append(conf)
    if idx in overlay_targets:
        overlays[idx] = (fr.copy(), y_abs, conf)
    idx += DETECT_STEP
cap.release()

times = np.array(times); ys = np.array(ys); confs = np.array(confs)
cms = y_to_cm(ys)

# ---------------------- SMOOTH ----------------------
cms_med = medfilt(cms, kernel_size=5)
cms_smooth = savgol_filter(cms_med, 11, 2, mode='nearest')

valid = times >= T_VALID_START
mask_baseline = (times >= T_VALID_START) & (times < T_VALID_START + 2.0)
baseline = float(np.median(cms_smooth[mask_baseline])) if mask_baseline.any() else float(cms_smooth[valid][0])
delta_h = cms_smooth - baseline
delta_V = delta_h * A_CM2

print(f'baseline level: {baseline:.2f} cm')
print(f'level range (smoothed): {cms_smooth[valid].min():.2f}..{cms_smooth[valid].max():.2f} cm '
      f'(swing {cms_smooth[valid].max()-cms_smooth[valid].min():.2f} cm)')
print(f'delta-V range: {delta_V[valid].min():.1f}..{delta_V[valid].max():.1f} mL '
      f'(swing {delta_V[valid].max()-delta_V[valid].min():.1f} mL)')

# ---------------------- PLOT ----------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax in axes:
    ax.axvspan(0, T_VALID_START, color='lightgrey', alpha=0.5, zorder=0)

axes[0].plot(times, cms, color='silver', lw=0.7, label='raw per-frame')
axes[0].plot(times[valid], cms_smooth[valid], color='C0', lw=2.2, label='smoothed')
axes[0].plot(times[~valid], cms_smooth[~valid], color='C0', lw=1.2, alpha=0.4)
axes[0].set_ylabel('Water level on ruler (cm)', fontsize=11)
axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title('CMR phantom: water level & ΔV vs time  (D = 9.0 cm)', fontsize=12)
axes[0].set_ylim(13.5, 19.0)

axes[1].plot(times[valid], delta_V[valid], color='C3', lw=2.2)
axes[1].plot(times[~valid], delta_V[~valid], color='C3', lw=1.2, alpha=0.4)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel(f'Δ Volume (mL)\nA = {A_CM2:.2f} cm²\nbaseline h = {baseline:.2f} cm',
                   fontsize=11)
axes[1].grid(alpha=0.3)

axes[2].plot(times, confs, color='C2', lw=1)
axes[2].axhline(0.05, color='k', lw=0.5, ls='--', alpha=0.5,
                label='low-confidence threshold')
axes[2].set_ylabel('Detector confidence', fontsize=11)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].grid(alpha=0.3); axes[2].legend(loc='lower right')

# stats text
vmin = float(delta_V[valid].min()); vmax = float(delta_V[valid].max())
hmin = float(cms_smooth[valid].min()); hmax = float(cms_smooth[valid].max())
sig = cms_smooth[valid] - cms_smooth[valid].mean()
ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
fs_est = 1.0 / (times[1] - times[0])
min_lag = int(2 * fs_est)
peak_lag = min_lag + int(np.argmax(ac[min_lag:int(15*fs_est)]))
period_s = peak_lag / fs_est
txt = (f'(t ≥ {T_VALID_START:.0f} s used)\n'
       f'level: {hmin:.2f} – {hmax:.2f} cm  (swing {hmax-hmin:.2f} cm)\n'
       f'ΔV:    {vmin:+.0f} – {vmax:+.0f} mL  (swing {vmax-vmin:.0f} mL)\n'
       f'period ≈ {period_s:.1f} s  ({60/period_s:.1f} cycles/min)')
axes[0].text(0.01, 0.97, txt, transform=axes[0].transAxes, va='top', ha='left',
             fontsize=10, family='monospace',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='C0'))

plt.tight_layout()
plt.savefig(OUT / 'level_and_volume.png', dpi=140)
print(f'period estimate: {period_s:.2f} s')

# ---------------------- CSV ----------------------
with open(OUT / 'level_volume.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'level_cm_raw', 'level_cm_smooth',
                'delta_h_cm', 'delta_V_mL', 'confidence'])
    for t, lr, ls_, dh, dv, c in zip(times, cms, cms_smooth, delta_h, delta_V, confs):
        w.writerow([f'{t:.3f}', f'{lr:.3f}', f'{ls_:.3f}',
                    f'{dh:.3f}', f'{dv:.2f}', f'{c:.3f}'])

# ---------------------- MONTAGE ----------------------
if overlays:
    keys = sorted(overlays.keys())
    imgs = []
    for k in keys:
        fr, y_abs, conf = overlays[k]
        vis = fr.copy()
        cv2.rectangle(vis, (RULER_X0, Y_TOP), (RULER_X1, Y_BOT), (255,0,0), 2)
        cv2.line(vis, (RULER_X0-40, y_abs), (RULER_X1+40, y_abs), (0,255,0), 3)
        t = k / fps
        cms_here = float(y_to_cm(y_abs))
        txt = f't={t:5.2f}s  level={cms_here:5.2f} cm  conf={conf:.2f}'
        cv2.putText(vis, txt, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        imgs.append(cv2.resize(vis, (vis.shape[1]//2, vis.shape[0]//2)))
    ncols = 3
    rows = []
    for i in range(0, len(imgs), ncols):
        row_imgs = imgs[i:i+ncols]
        while len(row_imgs) < ncols:
            row_imgs.append(np.zeros_like(imgs[0]))
        rows.append(np.hstack(row_imgs))
    cv2.imwrite(str(OUT / 'overlay_montage.jpg'), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---------------------- ANNOTATED VIDEO ----------------------
# For the annotated video we want a value at *every* frame, so we interpolate
# the smoothed level series (which is sparse at DETECT_STEP) back to per-frame.
all_t = np.arange(n_total) / fps
level_per_frame = np.interp(all_t, times, cms_smooth)
deltaV_per_frame = (level_per_frame - baseline) * A_CM2
ys_per_frame = np.interp(all_t, times, ys)            # pixel-y of the meniscus
conf_per_frame = np.interp(all_t, times, confs)

# build a fresh capture, write annotated frames to mp4
cap = cv2.VideoCapture(str(VIDEO_IN))
out_W = int(W * ANNOTATED_SCALE); out_H = int(H * ANNOTATED_SCALE)
fourcc = cv2.VideoWriter_fourcc(*ANNOTATED_FOURCC)
writer = cv2.VideoWriter(str(ANNOTATED_OUT), fourcc, fps, (out_W, out_H))
assert writer.isOpened(), f'failed to open writer for {ANNOTATED_OUT}'

# pre-render the plot strip (level vs time) once and re-use it
strip_h = int(out_H * 0.22)
strip_w = out_W
plot_fig, plot_ax = plt.subplots(figsize=(strip_w/100, strip_h/100), dpi=100)
plot_ax.plot(times[valid], cms_smooth[valid], color='C0', lw=1.8)
plot_ax.set_ylim(13.5, 19.0)
plot_ax.set_xlim(0, all_t[-1])
plot_ax.set_ylabel('level (cm)', fontsize=8)
plot_ax.tick_params(labelsize=8)
plot_ax.grid(alpha=0.3)
plot_ax.axvspan(0, T_VALID_START, color='lightgrey', alpha=0.4)
plt.tight_layout(pad=0.4)
plot_fig.canvas.draw()
strip_bg = np.asarray(plot_fig.canvas.buffer_rgba())[..., :3][:, :, ::-1].copy()
plt.close(plot_fig)
strip_bg = cv2.resize(strip_bg, (strip_w, strip_h))

print(f'rendering annotated video -> {ANNOTATED_OUT.name} '
      f'({out_W}x{out_H} @ {fps:.1f}fps, {n_total} frames)...')
for i in range(n_total):
    ok, fr = cap.read()
    if not ok: break

    y_abs = int(round(ys_per_frame[i]))
    cms_here = float(level_per_frame[i])
    dv_here = float(deltaV_per_frame[i])
    cf = float(conf_per_frame[i])
    t = i / fps

    # draw overlay on full-res frame, then resize
    cv2.rectangle(fr, (RULER_X0, Y_TOP), (RULER_X1, Y_BOT), (255,0,0), 2)
    cv2.line(fr, (RULER_X0-50, y_abs), (RULER_X1+50, y_abs), (0,255,0), 3)
    # small tick label of cm on the line
    cv2.putText(fr, f'{cms_here:.2f} cm',
                (RULER_X1+60, y_abs+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # text block (cv2.putText is ASCII-only -> avoid unicode)
    color = (255,255,255) if t >= T_VALID_START else (200,200,200)
    pre = '' if t >= T_VALID_START else '(dye mixing) '
    lines = [
        f't = {t:5.2f} s',
        f'{pre}level = {cms_here:5.2f} cm',
        f'{pre}dV    = {dv_here:+7.1f} mL',
        f'D = {D_CM:.1f} cm   A = {A_CM2:.2f} cm^2',
    ]
    box_h = 28 * len(lines) + 16
    cv2.rectangle(fr, (20, 20), (560, 20+box_h), (0,0,0), -1)
    cv2.rectangle(fr, (20, 20), (560, 20+box_h), color, 2)
    for k, line in enumerate(lines):
        cv2.putText(fr, line, (32, 56 + 28*k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    fr_small = cv2.resize(fr, (out_W, out_H))

    # paste the level-vs-time strip in the bottom-left, with a moving marker
    x_marker = int(t / all_t[-1] * strip_w)
    strip = strip_bg.copy()
    cv2.line(strip, (x_marker, 0), (x_marker, strip_h-1), (0,200,0), 2)
    fr_small[out_H-strip_h:, 0:strip_w] = cv2.addWeighted(
        fr_small[out_H-strip_h:, 0:strip_w], 0.0, strip, 1.0, 0)

    writer.write(fr_small)

cap.release()
writer.release()
print(f'wrote {ANNOTATED_OUT}  size={ANNOTATED_OUT.stat().st_size/1e6:.1f} MB')
print('done')
