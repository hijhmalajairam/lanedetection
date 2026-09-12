import numpy as np
import cv2

class LaneLine:
    def __init__(self):
        self.detected = False
        self.best_fit = None  
        self.current_fit = None 
        self.allx = None
        self.ally = None
        
        # ADVANCED: Kalman Filter for tracking the 3 polynomial coefficients (A, B, C)
        self.kf = cv2.KalmanFilter(3, 3)
        self.kf.transitionMatrix = np.eye(3, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(3, dtype=np.float32)
        # Process noise: how fast we expect the curve to physically change
        self.kf.processNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
        # Measurement noise: how much we trust our visual detections
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        
        self.initialized = False
        
    def reset(self):
        self.detected = False
        self.initialized = False
        self.best_fit = None
        self.current_fit = None
        self.kf.errorCovPost = np.eye(3, dtype=np.float32) * 1.0
    
    def update(self, fit, allx, ally):
        if fit is not None:
            self.detected = True
            self.current_fit = fit
            self.allx = allx
            self.ally = ally
            
            measurement = np.array(fit, dtype=np.float32).reshape(3, 1)
            
            if not self.initialized:
                # Initialize Kalman state
                self.kf.statePre = measurement
                self.kf.statePost = measurement
                self.best_fit = fit
                self.initialized = True
            else:
                # Correct with new measurement
                self.kf.correct(measurement)
                # Predict next state
                predicted = self.kf.predict()
                self.best_fit = predicted.flatten()
        else:
            self.detected = False
            if self.initialized:
                # OCCLUSION HANDLING: If we lose the line, PREDICT its position using physics!
                predicted = self.kf.predict()
                self.best_fit = predicted.flatten()

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50
    
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 3) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 3) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
    left_fit = None
    right_fit = None
    
    if len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        
    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
        cv2.polylines(out_img, pts_left, False, (0, 255, 255), 4)
        cv2.polylines(out_img, pts_right, False, (0, 255, 255), 4)
        
    return left_fit, leftx, lefty, right_fit, rightx, righty, out_img
