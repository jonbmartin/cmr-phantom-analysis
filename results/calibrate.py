"""Interactive ruler calibration tool — matplotlib version.

Navigate to a frame where the ruler marks are clearly visible, then click
each mark when prompted.  Saves calibration.json, which is loaded
automatically by process_clicks.py and annotate.py.

Keys — navigation phase:
    ← / → (or a / d)   step one frame
    [ / ]               jump ±1 second
    Enter               start calibrating

Keys — calibration phase:
    click               record the mark and advance to the next
    z                   undo last click
    Enter / q           save and quit
"""
import cv2, numpy as np, json
import matplotlib.pyplot as plt
from pathlib import Path

HERE  = Path(__file__).resolve().parent
REPO  = HERE.parent
VIDEO = REPO / 'reshot.MOV'
OUT   = HERE / 'calibration.json'

# cm marks to calibrate, listed top → bottom
CAL_CM_TARGETS = [21, 20, 19, 18, 17, 16, 15, 14, 13]

CROP_X0, CROP_X1 = 900, 1500
CROP_Y0, CROP_Y1 = 150, 1080


def read_crop(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(fr[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1], cv2.COLOR_BGR2RGB)


def main():
    cap      = cv2.VideoCapture(str(VIDEO))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    state = {
        'frame_idx': 0,
        'phase':     'nav',   # 'nav' | 'cal'
        'cal_idx':   0,
        'clicks':    {},      # cm_value → full_frame_y
    }

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.tight_layout(pad=1.5)

    im = ax.imshow(read_crop(cap, 0), origin='upper')
    ax.axis('off')

    cursor_hline = ax.axhline(-1, color='cyan', lw=1, alpha=0.8, visible=False)
    cursor_label = ax.text(4, 0, '', color='cyan', fontsize=8, va='bottom',
                           bbox=dict(fc='black', alpha=0.5, pad=1))
    mark_lines = {}  # cm → Line2D
    mark_texts = {}  # cm → Text

    # ------------------------------------------------------------------ helpers
    def update_title():
        if state['phase'] == 'nav':
            t = state['frame_idx'] / fps
            ax.set_title(
                f"frame {state['frame_idx']}  t={t:.1f} s\n"
                f"← / → (a/d) = ±1 frame   [ / ] = ±1 s   Enter = start calibrating",
                fontsize=9)
        else:
            done, total = state['cal_idx'], len(CAL_CM_TARGETS)
            if done < total:
                ax.set_title(
                    f"Click on the  {CAL_CM_TARGETS[done]} cm  mark"
                    f"  ({done+1}/{total})\n"
                    f"z = undo last   Enter / q = save + quit",
                    fontsize=9, color='limegreen')
            else:
                ax.set_title("All marks recorded — press Enter or q to save.",
                             fontsize=9, color='limegreen')
        fig.canvas.draw_idle()

    def refresh_frame():
        crop = read_crop(cap, state['frame_idx'])
        if crop is not None:
            im.set_data(crop)
        update_title()

    def save_and_quit():
        plt.close(fig)

    # ------------------------------------------------------------------ events
    def on_motion(event):
        if event.inaxes != ax or event.ydata is None:
            return
        y = float(event.ydata)
        cursor_hline.set_ydata([y, y])
        cursor_hline.set_visible(True)
        cursor_label.set_position((4, max(y - 3, 0)))
        cursor_label.set_text(f'y_full = {int(round(y)) + CROP_Y0}')
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.ydata is None:
            return
        if state['phase'] != 'cal':
            return
        idx = state['cal_idx']
        if idx >= len(CAL_CM_TARGETS):
            return
        crop_y = int(round(event.ydata))
        full_y = crop_y + CROP_Y0
        cm     = CAL_CM_TARGETS[idx]
        state['clicks'][cm] = full_y
        print(f'  {cm:2d} cm  →  y_full = {full_y}')
        state['cal_idx'] += 1
        # draw the mark
        if cm in mark_lines:
            mark_lines.pop(cm).remove()
            mark_texts.pop(cm).remove()
        mark_lines[cm] = ax.axhline(crop_y, color='lime', lw=1.5, zorder=5)
        mark_texts[cm] = ax.text(4, max(crop_y - 4, 4), f'{cm} cm',
                                 color='lime', fontsize=8, zorder=6,
                                 bbox=dict(fc='black', alpha=0.6, pad=1))
        update_title()

    def on_key(event):
        k = event.key
        if k == 'q' or (k == 'enter' and state['phase'] == 'cal'):
            save_and_quit()
        elif k == 'enter' and state['phase'] == 'nav':
            state['phase'] = 'cal'
            print(f'\nCalibrating on frame {state["frame_idx"]}'
                  f'  (t = {state["frame_idx"]/fps:.1f} s)')
            print('Click on each ruler mark as prompted.\n')
            update_title()
        elif state['phase'] == 'nav':
            if k in ('left', 'a'):
                state['frame_idx'] = max(0, state['frame_idx'] - 1)
                refresh_frame()
            elif k in ('right', 'd'):
                state['frame_idx'] = min(n_frames - 1, state['frame_idx'] + 1)
                refresh_frame()
            elif k == '[':
                state['frame_idx'] = max(0, state['frame_idx'] - int(fps))
                refresh_frame()
            elif k == ']':
                state['frame_idx'] = min(n_frames - 1, state['frame_idx'] + int(fps))
                refresh_frame()
        elif state['phase'] == 'cal' and k == 'z' and state['cal_idx'] > 0:
            state['cal_idx'] -= 1
            cm = CAL_CM_TARGETS[state['cal_idx']]
            state['clicks'].pop(cm, None)
            if cm in mark_lines:
                mark_lines.pop(cm).remove()
                mark_texts.pop(cm).remove()
            print(f'  undo: removed {cm} cm')
            update_title()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_title()
    plt.show()
    cap.release()

    # ------------------------------------------------------------------ save
    if state['clicks']:
        ordered_cm = sorted(state['clicks'].keys(), reverse=True)
        ordered_y  = [state['clicks'][c] for c in ordered_cm]
        result = {'cal_cm': ordered_cm, 'cal_y': ordered_y}
        json.dump(result, open(OUT, 'w'), indent=2)
        print(f'\nSaved {len(ordered_cm)} calibration points to {OUT}')
        print('CAL_CM =', ordered_cm)
        print('CAL_Y  =', ordered_y)
        print('\nRun process_clicks.py to regenerate the analysis.')
    else:
        print('No calibration points recorded — calibration.json not written.')


if __name__ == '__main__':
    main()
