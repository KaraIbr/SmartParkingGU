import cv2
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
samples_dir = os.path.join(base, 'data', 'raw', 'video_samples')

for fname in ['2A_t0s.jpg', '2B_t0s.jpg', 'Monarch_t0s.jpg']:
    img = cv2.imread(os.path.join(samples_dir, fname))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f'=== {fname} ({w}x{h}) ===')

    edges = cv2.Canny(gray, 50, 150)

    grid_rows, grid_cols = 6, 8
    cell_h, cell_w = h // grid_rows, w // grid_cols
    print(f'Edge density grid ({grid_rows}x{grid_cols}):')
    for r in range(grid_rows):
        row_vals = []
        for c in range(grid_cols):
            cell = edges[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            density = cell.mean()
            row_vals.append(f'{density:5.1f}')
        joined = ' | '.join(row_vals)
        print(f'  Row {r}: [{joined}]')

    print()

    roi = gray[h//4:3*h//4, w//4:3*w//4]
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)
    peaks = []
    for i in range(10, 245):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > 500:
                peaks.append((i, float(hist_smooth[i])))
    peaks.sort(key=lambda x: -x[1])
    print('Brightness peaks:')
    for val, count in peaks[:8]:
        if val < 80:
            label = 'dark (asphalt/shadows)'
        elif val < 160:
            label = 'mid (cars/objects)'
        else:
            label = 'bright (lines/sky/highlight)'
        print(f'  Brightness {val:3d} ({label})')

    h_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=200, maxLineGap=30)
    if h_lines is not None:
        horizontal = [l for l in h_lines if abs(l[0][1] - l[0][3]) < 20]
        vertical = [l for l in h_lines if abs(l[0][0] - l[0][2]) < 20]
        print(f'Lines: {len(h_lines)} total, {len(horizontal)} horizontal, {len(vertical)} vertical')
        if horizontal:
            y_positions = sorted(set([l[0][1] for l in horizontal]))
            print(f'Horizontal Y positions (parking rows?): {y_positions[:20]}')

    thresh = cv2.adaptiveThreshold(gray, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_contours = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if 150 < bw < 1000 and 80 < bh < 600:
            car_contours.append((x, y, bw, bh))
    print(f'Car-sized blobs: {len(car_contours)}')
    for i, (x, y, bw, bh) in enumerate(car_contours[:15]):
        print(f'  Blob {i+1}: pos=({x},{y}) size={bw}x{bh} center=({x+bw//2},{y+bh//2})')

    print()

for fname in ['2A_t0s.jpg', '2B_t0s.jpg', 'Monarch_t0s.jpg']:
    img = cv2.imread(os.path.join(samples_dir, fname))
    small = cv2.resize(img, (960, 540))
    cv2.imwrite(os.path.join(samples_dir, f'thumb_{fname}'), small)
    print(f'Thumbnail saved: thumb_{fname}')
