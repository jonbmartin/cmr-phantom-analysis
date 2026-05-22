"""Manual meniscus annotation tool.

Steps one keyframe at a time (every 1 s by default).  Click on the meniscus.
Keys:
    click   record meniscus position and advance
    z       undo the current frame's click (or step back if none)
    s       skip this frame (no click)
    a / ←   previous frame
    d / →   next frame (also re-records nothing)
    q       save and quit

Output: clicks.json next to this file, with one entry per keyframe:
    { "frame": <int>, "t": <sec>, "click_y": <int or null> }
"""
import cv2, numpy as np, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIDEO = REPO / 'reshot.MOV'
OUT = HERE / 'clicks.json'

STEP_SECONDS = 0.5          # one keyframe per half-second (~130 frames)
DISPLAY_SCALE = 1.0         # show at native res
# Crop wide enough to see the ruler + cup walls for context, tall enough to
# cover the expected meniscus range.
CROP_X0, CROP_X1 = 900, 1500
CROP_Y0, CROP_Y1 = 150, 1080

# calibration (for live cm readout under the cursor)
CAL_CM = np.array([21, 20, 19, 18, 17, 16, 15, 14, 13], dtype=float)
CAL_Y  = np.array([62, 183, 322, 445, 658, 788, 892, 974, 1049], dtype=float)
y_to_cm = np.poly1d(np.polyfit(CAL_Y, CAL_CM, 2))

def main():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    keyframes = []
    t = 0.0
    while t <= duration + 1e-6:
        f = int(round(t * fps))
        if f >= n: f = n - 1
        keyframes.append(f)
        t += STEP_SECONDS
    print(f'{len(keyframes)} keyframes')

    # load existing clicks if any
    clicks = {}
    if OUT.exists():
        existing = json.load(open(OUT))
        for e in existing:
            clicks[e['frame']] = e.get('click_y', None)
        print(f'loaded {len(clicks)} existing clicks')

    # state
    state = {'i': 0, 'mouse_xy': None, 'last_click_full_y': None}

    def save():
        out = []
        for f in keyframes:
            out.append({'frame': int(f), 't': float(f / fps),
                        'click_y': (int(clicks[f]) if (f in clicks and clicks[f] is not None) else None)})
        json.dump(out, open(OUT, 'w'), indent=2)

    win = 'Meniscus annotation  (click to set, z=undo, s=skip, a/d=prev/next, q=save+quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _):
        # x, y are in display coords -> convert to full-frame coords
        full_x = int(round(x / DISPLAY_SCALE)) + CROP_X0
        full_y = int(round(y / DISPLAY_SCALE)) + CROP_Y0
        state['mouse_xy'] = (full_x, full_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            f = keyframes[state['i']]
            clicks[f] = full_y
            state['last_click_full_y'] = full_y
            save()                         # autosave after every click
            # advance
            if state['i'] < len(keyframes) - 1:
                state['i'] += 1
            state['mouse_xy'] = (full_x, full_y)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        i = state['i']
        f = keyframes[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        crop = fr[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].copy()
        # draw existing click for this frame
        if f in clicks and clicks[f] is not None:
            yfull = clicks[f]
            y_disp = int((yfull - CROP_Y0) * DISPLAY_SCALE)
            cv2.line(crop, (0, y_disp), (crop.shape[1] - 1, y_disp), (0, 255, 0), 2)
            cms = float(y_to_cm(yfull))
            cv2.putText(crop, f'recorded: {cms:.2f} cm',
                        (10, y_disp - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        # draw mouse crosshair + live cm readout
        if state['mouse_xy'] is not None:
            fx, fy = state['mouse_xy']
            y_disp = int((fy - CROP_Y0) * DISPLAY_SCALE)
            cv2.line(crop, (0, y_disp), (crop.shape[1] - 1, y_disp), (0, 200, 255), 1)
            cms = float(y_to_cm(fy))
            cv2.putText(crop, f'cursor: {cms:.2f} cm',
                        (10, max(20, y_disp - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)
        # status banner
        t_sec = f / fps
        marked = sum(1 for v in clicks.values() if v is not None)
        banner = f'frame {f}  t={t_sec:5.2f}s  ({i+1}/{len(keyframes)})   marked: {marked}/{len(keyframes)}'
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(crop, banner, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if DISPLAY_SCALE != 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1]*DISPLAY_SCALE),
                                     int(crop.shape[0]*DISPLAY_SCALE)))
        cv2.imshow(win, crop)
        # exit if user closed the window with the X button
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q') or k == 27:      # q or ESC
            break
        elif k == ord('z'):
            if f in clicks:
                del clicks[f]
                save()
            elif state['i'] > 0:
                state['i'] -= 1
        elif k == ord('s'):
            clicks[f] = None
            save()
            if state['i'] < len(keyframes) - 1:
                state['i'] += 1
        elif k == ord('a') or k == 81:    # left
            if state['i'] > 0:
                state['i'] -= 1
        elif k == ord('d') or k == 83:    # right
            if state['i'] < len(keyframes) - 1:
                state['i'] += 1

    cap.release()
    cv2.destroyAllWindows()
    save()
    print(f'saved {OUT}  ({sum(1 for v in clicks.values() if v is not None)} clicks recorded)')

if __name__ == '__main__':
    main()
