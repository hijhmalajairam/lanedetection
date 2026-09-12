import cv2
import numpy as np

def color_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    lower_white = np.array([0, 160, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([15, 30, 80], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    
    max_sobel = np.max(abs_sobelx)
    if max_sobel == 0: max_sobel = 1
        
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 15) & (scaled_sobel <= 120)] = 255

    combined = np.zeros_like(sxbinary)
    combined[(color_mask == 255) | (sxbinary == 255)] = 255
    return combined

def get_perspective_transform_matrices(img_shape, top_y=0.63, bottom_y=0.95, top_w=0.10):
    """
    Calculates homography matrices using DYNAMIC parameters from the Web UI!
    This allows the horizon to be calibrated interactively.
    """
    h, w = img_shape[:2]
    
    # Source points (Dynamic Trapezoid)
    src = np.float32([
        [w * (0.5 - top_w/2), h * top_y], # Top-left
        [w * (0.5 + top_w/2), h * top_y], # Top-right
        [w * 0.85, h * bottom_y],         # Bottom-right
        [w * 0.15, h * bottom_y]          # Bottom-left
    ])

    # Destination points (Rectangle)
    offset = w * 0.25
    dst = np.float32([
        [offset, 0],
        [w - offset, 0],
        [w - offset, h],
        [offset, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp_perspective(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
