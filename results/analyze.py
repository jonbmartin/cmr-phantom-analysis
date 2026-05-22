"""Analysis pipeline v2: robust meniscus detection using brightness gradient
+ redness, with a search window seeded from the previous detection.

Outputs:
  /tmp/cmr2/results/level_and_volume.png
  /tmp/cmr2/results/level_volume.csv
  /tmp/cmr2/results/overlay_montage.jpg
  /tmp/cmr2/results/kymograph.png  (level vs time visualised on the ruler)
"""
import cv2, numpy as np, csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

VIDEO = '/home/jonathanmartin/Documents/cmr_phantom_analysis/reshot.MOV'
OUT = Path('/tmp/cmr2/results'); OUT.mkdir(exist_ok=True, parents=True)

# ---- calibration (hand-identified cm-tick rows)
CM = np.array([21, 20, 19, 18, 17, 16, 15, 14, 13], dtype=float)
YR = np.array([62, 183, 322, 445, 658, 788, 892, 974, 1049], dtype=float)
coeff = np.polyfit(YR, CM, 2)
y_to_cm = np.poly1d(coeff)

RULER_X0, RULER_X1 = 1095, 1255
Y_TOP, Y_BOT = 40, 1070

# ---- detector
def detect_meniscus(fr, prev_y=None, search_margin=200):
    """Find the meniscus row (in full-frame coordinates).

    Strategy:
      1. ROI brightness V averaged across width -> 1D profile vs row.
      2. Subtract smooth baseline (savgol high-pass).
      3. Compute downward brightness STEP via centred negative gradient over a
         small window.  Strong dark-step rows score high.
      4. Add a redness term as well (helps when dye is strong).
      5. Restrict search to +/- search_margin around prev_y when available.
      6. Return row + confidence value.
    """
    roi = fr[Y_TOP:Y_BOT, RULER_X0:RULER_X1]
    f = roi.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    V = np.maximum(np.maximum(B, G), R)
    V_row = V.mean(axis=1)

    # smooth a bit
    V_s = savgol_filter(V_row, 21, 2)
    # downward step score = V_above - V_below at each row, with avg over 25 px
    # use a finite difference with a small offset
    OFF = 25
    pad = np.pad(V_s, OFF, mode='edge')
    above = pad[OFF*2:][:len(V_s)] if False else np.array([V_s[max(0, i-OFF):i].mean() if i>0 else V_s[0] for i in range(len(V_s))])
    below = np.array([V_s[i+1:min(len(V_s), i+OFF+1)].mean() if i+1<len(V_s) else V_s[-1] for i in range(len(V_s))])
    step = above - below   # positive at a bright->dark transition (meniscus)

    # redness step likewise
    redness = R - 0.5 * (G + B)
    red_row = redness.mean(axis=1)
    red_s = savgol_filter(red_row, 21, 2)
    below_red = np.array([red_s[i+1:min(len(red_s), i+OFF+1)].mean() if i+1<len(red_s) else red_s[-1] for i in range(len(red_s))])
    above_red = np.array([red_s[max(0,i-OFF):i].mean() if i>0 else red_s[0] for i in range(len(red_s))])
    red_step = below_red - above_red  # positive when below is redder

    # normalise & combine
    s = step / max(1.0, step.max())
    rs = red_step / max(1.0, red_step.max())
    score = 0.65 * np.clip(s, 0, None) + 0.35 * np.clip(rs, 0, None)

    # search window
    if prev_y is not None:
        lo = max(0, (prev_y - Y_TOP) - search_margin)
        hi = min(len(score), (prev_y - Y_TOP) + search_margin)
        mask = np.zeros_like(score, dtype=bool)
        mask[lo:hi] = True
        score = np.where(mask, score, -1.0)

    best = int(np.argmax(score))
    conf = float(score[best])
    return Y_TOP + best, conf, step, red_step

# ---- run
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
STEP = 3

times, ys, cms, confs = [], [], [], []
overlays = {}
prev_y = None
# pre-pick frames for overlay so they exist regardless of STEP
overlay_targets = set()
for t_target in [0.5, 5.0, 15.0, 30.0, 45.0, 60.0]:
    f_target = int(t_target * fps)
    # round to nearest multiple of STEP
    f_target = (f_target // STEP) * STEP
    overlay_targets.add(f_target)

idx = 0
while idx < n:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        idx += STEP; continue
    y_abs, conf, _, _ = detect_meniscus(fr, prev_y=prev_y, search_margin=180)

    # If confidence too low and we have prev, prefer prev
    if conf < 0.05 and prev_y is not None:
        y_abs = prev_y
        conf = 0.0
    else:
        prev_y = y_abs

    t = idx / fps
    times.append(t); ys.append(y_abs); cms.append(float(y_to_cm(y_abs))); confs.append(conf)

    if idx in overlay_targets:
        vis = fr.copy()
        cv2.rectangle(vis, (RULER_X0, Y_TOP), (RULER_X1, Y_BOT), (255,0,0), 2)
        cv2.line(vis, (RULER_X0-40, y_abs), (RULER_X1+40, y_abs), (0,255,0), 3)
        txt = f't={t:5.2f}s  level={cms[-1]:5.2f} cm  conf={conf:.2f}'
        cv2.putText(vis, txt, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        overlays[idx] = vis
    idx += STEP
cap.release()

times = np.array(times); ys = np.array(ys); cms = np.array(cms); confs = np.array(confs)

# ---- smoothing (median then savgol; preserve endpoints with edge-padding)
cms_med = medfilt(cms, kernel_size=5)
cms_smooth = savgol_filter(cms_med, window_length=11, polyorder=2, mode='nearest')

# ---- volume
D_cm = 8.0
A_cm2 = np.pi * (D_cm/2)**2  # 50.265

# Drop the dye-mixing settling period at the start where redness is unstable.
T_VALID_START = 10.0
valid = times >= T_VALID_START

# baseline = median of first 2 s of valid data (typically near a trough)
mask_baseline = (times >= T_VALID_START) & (times < T_VALID_START + 2.0)
baseline = float(np.median(cms_smooth[mask_baseline])) if mask_baseline.any() else float(cms_smooth[valid][0])
delta_h = cms_smooth - baseline
delta_V = delta_h * A_cm2

# absolute volume relative to a notional zero at cup bottom is unknown; we
# report relative change only.
print(f'baseline level: {baseline:.2f} cm')
print(f'level range (smoothed): {cms_smooth.min():.2f}..{cms_smooth.max():.2f} cm '
      f'(swing {cms_smooth.max()-cms_smooth.min():.2f} cm)')
print(f'delta-V range: {delta_V.min():.1f}..{delta_V.max():.1f} mL '
      f'(swing {delta_V.max()-delta_V.min():.1f} mL)')

# ---- plots
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
# show the trimmed-out region greyed so it's obvious
for ax in axes:
    ax.axvspan(0, T_VALID_START, color='lightgrey', alpha=0.5, zorder=0)

axes[0].plot(times, cms, color='silver', lw=0.7, label='raw per-frame')
axes[0].plot(times[valid], cms_smooth[valid], color='C0', lw=2.2, label='smoothed (median + Sav-Gol)')
axes[0].plot(times[~valid], cms_smooth[~valid], color='C0', lw=1.2, alpha=0.4)
axes[0].set_ylabel('Water level on ruler (cm)', fontsize=11)
axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title('Reshot phantom: water level & estimated volume change vs time', fontsize=12)
axes[0].set_ylim(13.5, 19.0)

axes[1].plot(times[valid], delta_V[valid], color='C3', lw=2.2)
axes[1].plot(times[~valid], delta_V[~valid], color='C3', lw=1.2, alpha=0.4)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel(f'Δ Volume (mL)\nA = π·(D/2)² = {A_cm2:.2f} cm²\nbaseline = {baseline:.2f} cm', fontsize=11)
axes[1].grid(alpha=0.3)

axes[2].plot(times, confs, color='C2', lw=1)
axes[2].axhline(0.05, color='k', lw=0.5, ls='--', alpha=0.5, label='low-confidence threshold')
axes[2].set_ylabel('Detector confidence', fontsize=11)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].grid(alpha=0.3); axes[2].legend(loc='lower right')

# summary stats in figure
vmin = float(delta_V[valid].min()); vmax = float(delta_V[valid].max())
hmin = float(cms_smooth[valid].min()); hmax = float(cms_smooth[valid].max())
# estimate period via autocorrelation of valid signal
sig = cms_smooth[valid] - cms_smooth[valid].mean()
ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
# find first peak after zero-lag (skip lags < 2 s)
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

# CSV
with open(OUT / 'level_volume.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'level_cm_raw', 'level_cm_smooth', 'delta_h_cm', 'delta_V_mL', 'confidence'])
    for t, lr, ls_, dh, dv, c in zip(times, cms, cms_smooth, delta_h, delta_V, confs):
        w.writerow([f'{t:.3f}', f'{lr:.3f}', f'{ls_:.3f}', f'{dh:.3f}', f'{dv:.2f}', f'{c:.3f}'])

# montage
if overlays:
    overlay_keys = sorted(overlays.keys())
    imgs = [overlays[k] for k in overlay_keys]
    h_img = [cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)) for img in imgs]
    ncols = 3
    rows = []
    for i in range(0, len(h_img), ncols):
        row_imgs = h_img[i:i+ncols]
        while len(row_imgs) < ncols:
            row_imgs.append(np.zeros_like(h_img[0]))
        rows.append(np.hstack(row_imgs))
    montage = np.vstack(rows)
    cv2.imwrite(str(OUT / 'overlay_montage.jpg'), montage, [cv2.IMWRITE_JPEG_QUALITY, 88])

print('done')
