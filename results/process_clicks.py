"""Consume manual clicks (clicks.json) -> level/volume series + annotated
video.  Run AFTER annotate.py.
"""
import cv2, numpy as np, json, csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIDEO = REPO / 'reshot.MOV'
CLICKS = HERE / 'clicks.json'

D_CM = 9.0
A_CM2 = np.pi * (D_CM/2)**2
T_VALID_START = 0.0           # use everything — clicks are trusted

# calibration (same as analyze.py)
# calibration — load from file if available, else use coded defaults
_CAL_FILE = HERE / 'calibration.json'
if _CAL_FILE.exists():
    _cal = json.load(open(_CAL_FILE))
    CAL_CM = np.array(_cal['cal_cm'], dtype=float)
    CAL_Y  = np.array(_cal['cal_y'],  dtype=float)
else:
    CAL_CM = np.array([21, 20, 19, 18, 17, 16, 15, 14, 13], dtype=float)
    CAL_Y  = np.array([62, 183, 322, 445, 658, 788, 892, 974, 1049], dtype=float)
y_to_cm = lambda y: np.interp(y, CAL_Y, CAL_CM)

# load clicks
if not CLICKS.exists():
    raise SystemExit(
        f'No clicks file at {CLICKS}.\n'
        f'Run the annotation tool first:  python3 {HERE}/annotate.py\n'
        f'Click the meniscus on each keyframe.  The tool autosaves on every '
        f'click, so you can stop and resume.')
clicks = json.load(open(CLICKS))
clicks = [c for c in clicks if c.get('click_y') is not None]
ts_click = np.array([c['t'] for c in clicks])
ys_click = np.array([c['click_y'] for c in clicks], dtype=float)
cms_click = y_to_cm(ys_click)
print(f'{len(clicks)} valid clicks; t range {ts_click.min():.2f} .. {ts_click.max():.2f} s')

# open video for fps/n
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Per-frame level by spline-smooth interpolation through clicks.
all_t = np.arange(n_total) / fps
# clamp range to clicks
mask_within = (all_t >= ts_click.min()) & (all_t <= ts_click.max())
# cubic via np.interp + light savgol smooth in time
cms_interp = np.interp(all_t, ts_click, cms_click)
# light low-pass smoothing
if len(cms_interp) > 25:
    cms_smooth = savgol_filter(cms_interp, 25, 3, mode='nearest')
else:
    cms_smooth = cms_interp
ys_interp = np.interp(all_t, ts_click, ys_click)

baseline = float(cms_smooth.min())
delta_h = cms_smooth - baseline
delta_V = delta_h * A_CM2

print(f'baseline = {baseline:.2f} cm')
print(f'level range: {cms_smooth.min():.2f} .. {cms_smooth.max():.2f} cm '
      f'(swing {cms_smooth.max()-cms_smooth.min():.2f} cm)')
print(f'ΔV range:   {delta_V.min():.1f} .. {delta_V.max():.1f} mL '
      f'(swing {delta_V.max()-delta_V.min():.1f} mL)')

# period estimate via autocorr
sig = cms_smooth - cms_smooth.mean()
ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
min_lag = int(2 * fps)
peak_lag = min_lag + int(np.argmax(ac[min_lag:int(15*fps)]))
period_s = peak_lag / fps
print(f'period ≈ {period_s:.2f} s ({60/period_s:.1f} cycles/min)')

# TTL events
TTL_FILE = HERE / 'ttl_events.json'
ttl_events = []
if TTL_FILE.exists():
    ttl_events = json.load(open(TTL_FILE)).get('events', [])
    print(f'Loaded {len(ttl_events)} TTL events')

# stability analysis
print('\n--- Stability Analysis ---')
prom = (cms_smooth.max() - cms_smooth.min()) * 0.3
min_dist = int(fps * 2)
peaks,   _ = find_peaks( cms_smooth, prominence=prom, distance=min_dist)
troughs, _ = find_peaks(-cms_smooth, prominence=prom, distance=min_dist)
peak_t,   peak_v   = all_t[peaks],   cms_smooth[peaks]
trough_t, trough_v = all_t[troughs], cms_smooth[troughs]
if len(peak_t) > 1:
    p2p = np.diff(peak_t)
    print(f'Peaks detected:          {len(peaks)}')
    print(f'Peak-to-peak interval:   {p2p.mean():.3f} ± {p2p.std():.3f} s'
          f'  (CV = {100*p2p.std()/p2p.mean():.1f}%)')
    print(f'Peak level:              {peak_v.mean():.3f} ± {peak_v.std():.3f} cm')
if len(trough_t) > 1:
    t2t = np.diff(trough_t)
    print(f'Trough-to-trough:        {t2t.mean():.3f} ± {t2t.std():.3f} s'
          f'  (CV = {100*t2t.std()/t2t.mean():.1f}%)')
    print(f'Trough level:            {trough_v.mean():.3f} ± {trough_v.std():.3f} cm')
print(f'ΔV mean ± std:           {delta_V.mean():.1f} ± {delta_V.std():.1f} mL')
if len(peaks) > 0:
    peak_dV = delta_V[peaks]
    print(f'Peak ΔV per cycle:       {peak_dV.mean():.1f} ± {peak_dV.std():.1f} mL'
          f'  (range {peak_dV.min():.1f}–{peak_dV.max():.1f} mL, n={len(peaks)})')
else:
    print(f'ΔV peak (max):           {delta_V.max():.1f} mL')
if ttl_events and len(ttl_events) > 1:
    ttl_ts = np.array([e['t'] for e in ttl_events])
    ttl_iv = np.diff(ttl_ts)
    print(f'TTL intervals:           {ttl_iv.mean():.3f} ± {ttl_iv.std():.3f} s'
          f'  (CV = {100*ttl_iv.std()/ttl_iv.mean():.1f}%)')
print()

# plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(all_t, cms_smooth, color='C0', lw=2, label='interpolated')
axes[0].plot(ts_click, cms_click, 'o', color='C0', ms=4, label='labeled')
axes[0].set_ylabel('phantom fluid level (cm)', fontsize=11)
axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')

axes[1].plot(all_t, delta_V, color='C3', lw=2)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('Δ Volume (mL)', fontsize=11)
axes[1].set_xlabel('Time (s)', fontsize=11); axes[1].grid(alpha=0.3)
axes[0].text(0.01, 0.97,
             f'level: {cms_smooth.min():.2f}–{cms_smooth.max():.2f} cm  '
             f'(swing {cms_smooth.max()-cms_smooth.min():.2f} cm)\n'
             f'ΔV:    {delta_V.min():+.0f}–{delta_V.max():+.0f} mL  '
             f'(swing {delta_V.max()-delta_V.min():.0f} mL)',

             transform=axes[0].transAxes, va='top', ha='left',
             fontsize=10, family='monospace',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='C0'))
for i, ev in enumerate(ttl_events):
    lbl = 'TTL' if i == 0 else None
    axes[0].axvline(ev['t'], color='red', lw=0.8, alpha=0.5, ls='--', label=lbl)
    axes[1].axvline(ev['t'], color='red', lw=0.8, alpha=0.5, ls='--')
if ttl_events:
    axes[0].legend(loc='upper right')
    dcms = np.gradient(cms_smooth, all_t)
    delays = []
    for ev in ttl_events:
        t_ttl = ev['t']
        mask = (all_t > t_ttl) & (all_t < t_ttl + 2.0)
        if not np.any(mask):
            continue
        d_win, t_win = dcms[mask], all_t[mask]
        p95 = float(np.percentile(d_win, 95)) if len(d_win) else 0.0
        thresh = max(0.02, 0.15 * p95)
        rising = t_win[d_win > thresh]
        if len(rising):
            delays.append(float(rising[0]) - t_ttl)
    if delays:
        delays = np.array(delays)
        mean_ms = delays.mean() * 1000
        std_ms  = delays.std()  * 1000
        print(f'Pneumatic delay: {mean_ms:.0f} +/- {std_ms:.0f} ms (n={len(delays)})')
plt.tight_layout()
plt.savefig(HERE / 'level_and_volume.png', dpi=140)

# CSV
with open(HERE / 'level_volume.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'level_cm', 'delta_h_cm', 'delta_V_mL'])
    for t, lv, dh, dv in zip(all_t, cms_smooth, delta_h, delta_V):
        w.writerow([f'{t:.3f}', f'{lv:.3f}', f'{dh:.3f}', f'{dv:.2f}'])

# annotated video
RULER_X0, RULER_X1 = 1095, 1255
OUT_MP4 = HERE / 'annotated.mp4'
SCALE = 0.6
out_W, out_H = int(W*SCALE), int(H*SCALE)

# preview strip
strip_h = int(out_H * 0.25); strip_w = out_W
plot_fig, plot_ax = plt.subplots(figsize=(strip_w/100, strip_h/100), dpi=100)
plot_ax.plot(all_t, cms_smooth, color='C0', lw=1.6)
plot_ax.plot(ts_click, cms_click, '.', color='C0', ms=3)
plot_ax.set_xlim(0, all_t[-1])
plot_ax.set_ylabel('level (cm)', fontsize=8)
plot_ax.tick_params(labelsize=8); plot_ax.grid(alpha=0.3)
for ev in ttl_events:
    plot_ax.axvline(ev['t'], color='red', lw=0.8, alpha=0.6, ls='--')
plt.tight_layout(pad=0.4)
plot_fig.canvas.draw()
strip_bg = np.asarray(plot_fig.canvas.buffer_rgba())[..., :3][:, :, ::-1].copy()
# Capture the axes pixel bounds so the time marker aligns with the data
ax_pos = plot_ax.get_position()   # figure fractions
ax_x0_px = ax_pos.x0 * strip_w
ax_x1_px = (ax_pos.x0 + ax_pos.width) * strip_w
t_min, t_max = plot_ax.get_xlim()
plt.close(plot_fig)
strip_bg = cv2.resize(strip_bg, (strip_w, strip_h))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUT_MP4), fourcc, fps, (out_W, out_H + strip_h))

print(f'rendering annotated -> {OUT_MP4.name} ...')
cap = cv2.VideoCapture(str(VIDEO))
for i in range(n_total):
    ok, fr = cap.read()
    if not ok: break
    y_abs = int(round(ys_interp[i]))
    cms_here = float(cms_smooth[i])
    dv_here = float(delta_V[i])
    t = i / fps

    cv2.line(fr, (RULER_X0-60, y_abs), (RULER_X1+60, y_abs), (0,255,0), 3)
    cv2.putText(fr, f'{cms_here:.2f} cm',
                (RULER_X1+70, y_abs+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    lines = [
        f't = {t:5.2f} s',
        f'level = {cms_here:5.2f} cm',
        f'dV    = {dv_here:+7.1f} mL',
        f'D = {D_CM:.1f} cm   A = {A_CM2:.2f} cm^2  (manual)',
    ]
    box_h = 28 * len(lines) + 16
    cv2.rectangle(fr, (20, 20), (620, 20+box_h), (0,0,0), -1)
    cv2.rectangle(fr, (20, 20), (620, 20+box_h), (255,255,255), 2)
    for k, line in enumerate(lines):
        cv2.putText(fr, line, (32, 56 + 28*k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    fr_small = cv2.resize(fr, (out_W, out_H))
    x_frac = (t - t_min) / (t_max - t_min)
    x_marker = int(ax_x0_px + x_frac * (ax_x1_px - ax_x0_px))
    x_marker = max(int(ax_x0_px), min(int(ax_x1_px), x_marker))
    strip = strip_bg.copy()
    cv2.line(strip, (x_marker, 0), (x_marker, strip_h-1), (0,200,0), 2)
    frame_out = np.vstack([fr_small, strip])
    writer.write(frame_out)
cap.release(); writer.release()
print(f'wrote {OUT_MP4} ({OUT_MP4.stat().st_size/1e6:.1f} MB)')
print('done')
