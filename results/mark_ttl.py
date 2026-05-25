"""Mark TTL trigger events by stepping through the video and pressing t or space.

Saves ttl_events.json.  Re-run process_clicks.py afterwards to see events
overlaid on the level timecourse and get the pneumatic delay estimate.

Keys:
    left / right (a / d)   +-1 frame
    [ / ]                  +-1 second
    space / t              mark TTL ON at current frame
    z                      undo last mark
    Enter / q              save and quit
"""
import cv2, numpy as np, json, csv
import matplotlib.pyplot as plt
from pathlib import Path

HERE  = Path(__file__).resolve().parent
REPO  = HERE.parent
VIDEO = REPO / 'reshot.MOV'
OUT   = HERE / 'ttl_events.json'


def load_level():
    csv_path = HERE / 'level_volume.csv'
    if not csv_path.exists():
        return None, None
    times, levels = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row['time_s']))
            levels.append(float(row['level_cm']))
    return np.array(times), np.array(levels)


def main():
    cap      = cv2.VideoCapture(str(VIDEO))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # load existing marks (insertion order preserved)
    events = []
    if OUT.exists():
        events = json.load(open(OUT)).get('events', [])
        print(f'Loaded {len(events)} existing TTL events')

    all_t, levels = load_level()
    state = {'frame_idx': 0}

    scale  = min(1.0, 1200 / W)
    disp_w = int(W * scale)
    disp_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (disp_w, disp_h)) if scale != 1.0 else rgb

    has_level = all_t is not None
    ttl_lines = []   # parallel to events list

    if has_level:
        fig = plt.figure(figsize=(13, 8))
        ax_img = fig.add_axes([0.0,  0.28, 1.0, 0.72])
        ax_lvl = fig.add_axes([0.07, 0.05, 0.91, 0.20])
        ax_lvl.plot(all_t, levels, color='C0', lw=1.5)
        ax_lvl.set_xlim(0, all_t[-1])
        ax_lvl.set_xlabel('Time (s)', fontsize=8)
        ax_lvl.set_ylabel('Level (cm)', fontsize=8)
        ax_lvl.tick_params(labelsize=7)
        ax_lvl.grid(alpha=0.3)
        t_vline = ax_lvl.axvline(0, color='orange', lw=1.5)
        for e in events:
            ttl_lines.append(ax_lvl.axvline(e['t'], color='red', lw=1, alpha=0.7))
    else:
        fig, ax_img = plt.subplots(figsize=(13, 7))
        ax_lvl  = None
        t_vline = None

    ax_img.axis('off')
    im = ax_img.imshow(read_frame(0))

    def redraw():
        t = state['frame_idx'] / fps
        ax_img.set_title(
            f"frame {state['frame_idx']}   t = {t:.3f} s   ({len(events)} TTL marks)\n"
            f"left/right (a/d) = +-1 frame   [ / ] = +-1 s   "
            f"space/t = mark TTL   z = undo   Enter/q = save+quit",
            fontsize=8)
        if t_vline is not None:
            t_vline.set_xdata([t, t])
        fig.canvas.draw_idle()

    def go(idx):
        state['frame_idx'] = max(0, min(n_frames - 1, idx))
        fr = read_frame(state['frame_idx'])
        if fr is not None:
            im.set_data(fr)
        redraw()

    def mark():
        t = round(state['frame_idx'] / fps, 6)
        if any(abs(t - e['t']) < 1 / fps for e in events):
            print(f'  Already marked near t={t:.3f}s'); return
        events.append({'frame': state['frame_idx'], 't': t})
        print(f'  Marked TTL at t={t:.3f}s  ({len(events)} total)')
        if ax_lvl is not None:
            ttl_lines.append(ax_lvl.axvline(t, color='red', lw=1, alpha=0.7))
        redraw()

    def undo():
        if not events:
            return
        e = events.pop()
        print(f'  Removed TTL at t={e["t"]:.3f}s')
        if ttl_lines:
            ttl_lines.pop().remove()
        redraw()

    def on_key(event):
        k, idx = event.key, state['frame_idx']
        if   k in ('q', 'enter'):    plt.close(fig)
        elif k in ('left',  'a'):    go(idx - 1)
        elif k in ('right', 'd'):    go(idx + 1)
        elif k == '[':               go(idx - int(fps))
        elif k == ']':               go(idx + int(fps))
        elif k in (' ', 't'):        mark()
        elif k == 'z':               undo()

    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    cap.release()

    if events:
        sorted_ev = sorted(events, key=lambda e: e['t'])
        json.dump({'events': sorted_ev}, open(OUT, 'w'), indent=2)
        print(f'\nSaved {len(sorted_ev)} TTL events to {OUT}')
        ts = np.array([e['t'] for e in sorted_ev])
        if len(ts) > 1:
            iv = np.diff(ts)
            print(f'Inter-TTL interval: {iv.mean():.3f} +/- {iv.std():.3f} s')
    else:
        print('No TTL events recorded.')


if __name__ == '__main__':
    main()
