import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_sad(left, right, window_size):
    height, width = left.shape
    disparity_map = np.zeros((height, width), np.uint8)
    offset = window_size // 2
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            best_offset = 0
            min_sad = float('inf')
            for d in range(x):
                if x - d - offset < 0:
                    continue
                sad = np.sum(np.abs(
                    left[y-offset:y+offset+1, x-offset:x+offset+1] -
                    right[y-offset:y+offset+1, (x-d)-offset:(x-d)+offset+1]
                ))
                if sad < min_sad:
                    min_sad = sad
                    best_offset = d
            disparity_map[y, x] = int(np.clip(best_offset, 0, 255))

    return disparity_map

def compute_ssd(left, right, window_size):
    height, width = left.shape
    disparity_map = np.zeros((height, width), np.uint8)
    offset = window_size // 2
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            best_offset = 0
            min_ssd = float('inf')
            for d in range(x):
                if x - d - offset < 0:
                    continue
                ssd = np.sum((
                    left[y-offset:y+offset+1, x-offset:x+offset+1] -
                    right[y-offset:y+offset+1, (x-d)-offset:(x-d)+offset+1]
                )**2)
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = d
            disparity_map[y, x] = np.clip(best_offset, 0, 255)

    return disparity_map

def dynamic_programming(left_line, right_line, sigma=2, c0=1):
    n = len(left_line)
    D = np.zeros((n+1, n+1))
    D[0, 1:] = np.arange(1, n+1) * c0
    D[1:, 0] = np.arange(1, n+1) * c0
    
    for i in range(1, n+1):
        for j in range(1, n+1):
            cost = ((left_line[i-1] - right_line[j-1])**2) / (sigma**2)
            D[i, j] = min(
                D[i-1, j-1] + cost,
                D[i-1, j] + c0,
                D[i, j-1] + c0
            )
    return D

def backtrack(D):
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = []
    while i > 0 and j > 0:
        if D[i, j] == D[i-1, j-1] + ((0)**2)/4:
            path.append((i-1, j-1))
            i -= 1
            j -= 1
        elif D[i, j] == D[i-1, j] + 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]

def plot_alignment(path, left_line, right_line):
    plt.figure(figsize=(8,8))
    plt.plot([0], [0], 'go')
    x, y = 0, 0
    for i, j in path:
        if i == x+1 and j == y+1:
            plt.plot([y, j], [x, i], 'b-')
        elif i == x+1:
            plt.plot([y, y], [x, i], 'r-')
        else:
            plt.plot([y, j], [x, x], 'g-')
        x, y = i, j
    plt.title('Alignment Path')
    plt.xlabel('Right Image')
    plt.ylabel('Left Image')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    # Load grayscale images
    left = cv2.imread('stereo_materials/l1.png', cv2.IMREAD_GRAYSCALE)
    right = cv2.imread('stereo_materials/r1.png', cv2.IMREAD_GRAYSCALE)

    if left is None or right is None:
        print("Error: Could not load images.")
        return

    window_sizes = [1, 5, 9]

    for w in window_sizes:
        sad_map = compute_sad(left, right, w)
        ssd_map = compute_ssd(left, right, w)

        cv2.imwrite(f'sad_disparity_w{w}.png', sad_map)
        cv2.imwrite(f'ssd_disparity_w{w}.png', ssd_map)

    # Dynamic programming on one scanline (example)
    scanline = 100
    left_line = left[scanline]
    right_line = right[scanline]

    D = dynamic_programming(left_line, right_line)
    path = backtrack(D)

    plot_alignment(path, left_line, right_line)

if __name__ == "__main__":
    main()
