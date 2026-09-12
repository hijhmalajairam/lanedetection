import cv2
import numpy as np
from src.vision_utils import color_threshold, get_perspective_transform_matrices, warp_perspective
from src.lane_tracker import LaneLine, fit_polynomial

class LDWSPipeline:
    def __init__(self):
        self.left_line = LaneLine()
        self.right_line = LaneLine()
        self.M = None
        self.Minv = None
        
        self.ym_per_pix = 30 / 720  
        self.xm_per_pix = 3.7 / 700 
        
        # Web UI State Variables
        self.turn_signal = None
        self.show_dashboard = True
        self.top_y = 0.63
        self.bottom_y = 0.95
        self.top_w = 0.10
        self.auto_calib = True
        self.view_mode = 'final'
        self.smooth_steering_angle = 0.0  
        self.prev_gray = None
        self.fcw_active = False
        self.ldws_active = False

        # ADAS Machine Learning Models
        self.left_line = LaneLine()
        self.right_line = LaneLine()
        self.turn_signal = None 
        try:
            self.car_cascade = cv2.CascadeClassifier('cars.xml')
            self.sign_cascade = cv2.CascadeClassifier('stop_data.xml')
        except AttributeError:
            self.car_cascade = None
            self.sign_cascade = None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # Always recalculate M based on dynamic Web UI sliders (solves Flask threading race condition)
        self.M, self.Minv = get_perspective_transform_matrices(frame.shape, self.top_y, self.bottom_y, self.top_w)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_thresh = color_threshold(frame)
        binary_warped = warp_perspective(binary_thresh, self.M)
        
        signs = []
        if hasattr(self, 'sign_cascade') and self.sign_cascade is not None:
            signs = self.sign_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        
        left_fit, leftx, lefty, right_fit, rightx, righty, out_img = fit_polynomial(binary_warped)
        self.left_line.update(left_fit, leftx, lefty)
        self.right_line.update(right_fit, rightx, righty)
        
        result = self.draw_lane(frame, binary_warped)
        result, radar = self.add_metrics_dashboard_fcw(result, binary_thresh, binary_warped, out_img, signs)
        
        ret_frame = result
        if self.view_mode == 'roi':
            roi_img = np.zeros_like(frame)
            roi_pts = np.array([
                [int(w * (0.5 - self.top_w/2)), int(h * self.top_y)],
                [int(w * (0.5 + self.top_w/2)), int(h * self.top_y)],
                [int(w * 0.95), int(h * self.bottom_y)],
                [int(w * 0.05), int(h * self.bottom_y)]
            ], dtype=np.int32)
            cv2.fillConvexPoly(roi_img, roi_pts, (0, 255, 0))
            ret_frame = cv2.addWeighted(frame, 0.7, roi_img, 0.3, 0)
            cv2.polylines(ret_frame, [roi_pts], True, (0, 255, 255), 3)
            cv2.putText(ret_frame, "Region of Interest (ROI)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
        elif self.view_mode == 'edge':
            ret_frame = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
            
        elif self.view_mode == 'hough':
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            # Mask to ROI
            roi_pts = np.array([
                [int(w * (0.5 - self.top_w/2)), int(h * self.top_y)],
                [int(w * (0.5 + self.top_w/2)), int(h * self.top_y)],
                [int(w * 0.95), int(h * self.bottom_y)],
                [int(w * 0.05), int(h * self.bottom_y)]
            ], dtype=np.int32)
            mask = np.zeros_like(edges)
            cv2.fillConvexPoly(mask, roi_pts, 255)
            masked = cv2.bitwise_and(edges, mask)
            # Hough Transform
            lines = cv2.HoughLinesP(masked, 2, np.pi/180, 30, minLineLength=30, maxLineGap=200)
            hough_overlay = frame.copy()
            line_count = 0
            if lines is not None:
                for line in lines:
                    coords = line.flatten()
                    if len(coords) == 4:
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        cv2.line(hough_overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        line_count += 1
            cv2.putText(hough_overlay, f"Hough Lines Detected: {line_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ret_frame = hough_overlay
            
        elif self.view_mode == 'warp':
            ret_frame = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
            
        elif self.view_mode == 'sliding':
            ret_frame = out_img
            
        elif self.view_mode == 'flow':
            flow_img = frame.copy()
            if self.prev_gray is not None:
                p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                if p0 is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
                    if p1 is not None:
                        good_new = p1[st==1]
                        good_old = p0[st==1]
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            flow_img = cv2.line(flow_img, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
                            flow_img = cv2.circle(flow_img, (int(a), int(b)), 4, (0, 0, 255), -1)
            ret_frame = flow_img
            
        elif self.view_mode == 'radar':
            ret_frame = cv2.resize(radar, (w, h))
            
        elif self.view_mode == 'clean':
            ret_frame = self.draw_lane(frame, binary_warped)
            
        self.prev_gray = gray.copy()
        return ret_frame
        
    def draw_lane(self, undist, warped):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Turn signal visualization
            if self.turn_signal == 'left':
                cv2.fillPoly(color_warp, np.int_([pts_left]), (0, 255, 255))
            elif self.turn_signal == 'right':
                cv2.fillPoly(color_warp, np.int_([pts_right]), (0, 255, 255))
            else:
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = warp_perspective(color_warp, self.Minv)
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def draw_steering_wheel(self, img, angle):
        h, w = img.shape[:2]
        r = max(30, min(60, h // 12))
        cx, cy = r + 30, h - r - 30
        
        cv2.circle(img, (cx, cy), r, (150, 150, 150), 12)
        cv2.circle(img, (cx, cy), r-6, (50, 50, 50), 2)
        
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        lx, ly = int(cx - r * cos_t), int(cy - r * sin_t)
        rx, ry = int(cx + r * cos_t), int(cy + r * sin_t)
        bx, by = int(cx + r * sin_t), int(cy - r * cos_t)
        
        cv2.line(img, (cx, cy), (lx, ly), (150, 150, 150), 10)
        cv2.line(img, (cx, cy), (rx, ry), (150, 150, 150), 10)
        cv2.line(img, (cx, cy), (bx, by), (150, 150, 150), 10)
        
        cv2.circle(img, (cx, cy), 15, (255, 100, 100), -1)
        cv2.putText(img, "LKAS", (cx - 20, cy - r - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def add_metrics_dashboard_fcw(self, frame, binary_thresh, binary_warped, sliding_window_img, signs):
        h, w = frame.shape[:2]
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        
        dash_w, dash_h = 320, 180
        radar = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        cv2.rectangle(radar, (0, 0), (dash_w, dash_h), (30, 30, 30), -1) # Dark grey background
        
        # Draw fake lanes on radar for context
        cv2.line(radar, (dash_w//2 - 50, 0), (dash_w//2 - 50, dash_h), (150, 150, 150), 2)
        cv2.line(radar, (dash_w//2 + 50, 0), (dash_w//2 + 50, dash_h), (150, 150, 150), 2)
        
        # Ego Car (White)
        ego_x, ego_y = dash_w//2, dash_h - 20
        cv2.rectangle(radar, (ego_x - 10, ego_y - 20), (ego_x + 10, ego_y + 10), (255, 255, 255), -1)
        cv2.rectangle(radar, (ego_x - 10, ego_y - 20), (ego_x + 10, ego_y + 10), (0, 255, 255), 1)
        
        fcw_warning = False
        
        # Process Traffic Signs (safely)
        try:
            if len(signs) > 0:
                for (x, y, w_box, h_box) in signs:
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 255), 3)
                    cv2.putText(frame, "STOP SIGN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception:
            pass

        # Deep Learning Vehicle Detection (MobileNet-SSD)
        if hasattr(self, 'net') and self.net is not None:
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # MobileNet Classes we care about: 2 (bicycle), 6 (bus), 7 (car), 14 (motorbike), 15 (person)
            TARGET_CLASSES = {2: 'Bike', 6: 'Bus', 7: 'Car', 14: 'Motorcycle', 15: 'Pedestrian'}
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.25:
                    idx = int(detections[0, 0, i, 1])
                    if idx in TARGET_CLASSES:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        w_box = endX - startX
                        h_box = endY - startY
                        x = startX
                        y = startY
                        
                        if y + h_box < h * 0.4: continue
                        if w_box < 10 or h_box < 10: continue
                        
                        # Distance estimation
                        dist_m = (1.8 * 800) / max(w_box, 1)
                        label = f"{TARGET_CLASSES[idx]} {dist_m:.1f}m"
                        
                        color_box = (0, 0, 255) if dist_m < 20 else (255, 150, 0)
                        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color_box, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                        
                        car_center_x = x + w_box/2
                        # Balanced lane detection
                        in_lane = (w//2 - 250 < car_center_x < w//2 + 250)
                        # Balanced distance sensitivity
                        if dist_m < 35 and in_lane:
                            fcw_warning = True
                            
                        # Plot on Top-Down Radar Map
                        radar_y = max(10, min(int(dash_h - (dist_m / 100.0) * dash_h), dash_h - 30))
                        
                        x_offset_pct = (car_center_x - w/2) / (w/2)
                        radar_x = int(ego_x + (x_offset_pct * (dash_w/2)))
                        radar_x = max(10, min(radar_x, dash_w - 10))
                        
                        color_radar = (0, 0, 255) if (dist_m < 20 and in_lane) else (0, 150, 255)
                        cv2.rectangle(radar, (radar_x - 10, radar_y - 15), (radar_x + 10, radar_y + 15), color_radar, -1)

        # Draw Warning overlay and set state
        import time
        if fcw_warning:
            self._fcw_until = time.time() + 1.0
            cv2.putText(frame, "! FORWARD COLLISION WARNING !", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        self.fcw_active = getattr(self, '_fcw_until', 0) > time.time()

        if left_fit is not None and right_fit is not None:
            y_eval = h
            left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            
            lane_center = (left_x + right_x) / 2.0
            car_center = w / 2.0
            offset_m = (car_center - lane_center) * self.xm_per_pix
            
            left_curverad = ((1 + (2*left_fit[0]*y_eval*self.ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
            right_curverad = ((1 + (2*right_fit[0]*y_eval*self.ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
            curve_rad = (left_curverad + right_curverad) / 2.0
            
            if abs(offset_m) > 1.5:
                self.left_line.reset()
                self.right_line.reset()

            warning = abs(offset_m) > 0.35
            if (offset_m > 0 and self.turn_signal == 'right') or (offset_m < 0 and self.turn_signal == 'left'):
                warning = False
            
            raw_angle = (left_fit[0] + right_fit[0]) * 50000 + (offset_m * 45)
            self.smooth_steering_angle = 0.8 * self.smooth_steering_angle + 0.2 * raw_angle
            self.draw_steering_wheel(frame, self.smooth_steering_angle)

            # Draw Radar Lanes
            scale_x, scale_y = dash_w / w, dash_h / h
            ploty_orig = np.linspace(0, h-1, dash_h)
            l_x = (left_fit[0]*ploty_orig**2 + left_fit[1]*ploty_orig + left_fit[2]) * scale_x
            r_x = (right_fit[0]*ploty_orig**2 + right_fit[1]*ploty_orig + right_fit[2]) * scale_x
            ploty_radar = np.linspace(0, dash_h-1, dash_h)
            pts_l = np.int32(np.column_stack((l_x, ploty_radar)))
            pts_r = np.int32(np.column_stack((r_x, ploty_radar)))
            cv2.polylines(radar, [pts_l], False, (0, 255, 0), 2)
            cv2.polylines(radar, [pts_r], False, (0, 255, 0), 2)

            direction = "Right" if offset_m > 0 else "Left"
            fs = max(0.4, h / 800)  # Scale font with resolution
            th = max(1, int(fs * 2))
            
            # 1. Calculate Drift Angle (Heading) in degrees
            # Derivative of the polynomial at the bottom of the screen (y = h) gives the tangent slope
            left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
            right_slope = 2 * right_fit[0] * y_eval + right_fit[1]
            avg_slope = (left_slope + right_slope) / 2.0
            drift_angle_deg = np.degrees(np.arctan(avg_slope))
            
            # 2. Calculate Confidence Score (%)
            # Confidence drops if the car is way off center, or if the curves don't match
            curve_diff = abs(left_curverad - right_curverad) / max(left_curverad, right_curverad, 1)
            confidence = 100.0 - (abs(offset_m) * 15.0) - (curve_diff * 20.0)
            confidence = max(15.0, min(99.9, confidence)) # Clamp between 15% and 99.9%
            
            # Draw Metrics
            y_base = int(h*0.06)
            y_step = int(h*0.06)
            cv2.putText(frame, f"Tracking Confidence: {confidence:.1f}%", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0) if confidence > 80 else (0,165,255), th)
            cv2.putText(frame, f"Offset: {abs(offset_m):.2f}m {direction}", (20, y_base + y_step), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
            cv2.putText(frame, f"Drift Angle: {abs(drift_angle_deg):.1f} deg {'Right' if drift_angle_deg > 0 else 'Left'}", (20, y_base + y_step*2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
            cv2.putText(frame, f"Curve Radius: {curve_rad:.0f}m", (20, y_base + y_step*3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
            
            if self.turn_signal:
                cv2.putText(frame, f"[ {self.turn_signal.upper()} TURN SIGNAL ON ]", (20, y_base + y_step*4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th)
            
            import time
            if warning:
                self._ldws_until = time.time() + 1.0
                cv2.putText(frame, "LANE DEPARTURE WARNING!", (20, y_base + y_step*5), cv2.FONT_HERSHEY_SIMPLEX, fs*1.2, (0, 0, 255), th+2)
            self.ldws_active = getattr(self, '_ldws_until', 0) > time.time()

        if self.view_mode == 'final':
            dash_w = min(320, w // 4)
            dash_h = min(180, (h - 40) // 3)
            
            if dash_w > 50 and dash_h > 30:
                radar_resized = cv2.resize(radar, (dash_w, dash_h))
                cv2.rectangle(radar_resized, (0,0), (dash_w-1, dash_h-1), (255,255,255), 2)
                
                thumb1 = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2BGR)
                thumb1 = cv2.resize(thumb1, (dash_w, dash_h))
                cv2.rectangle(thumb1, (0,0), (dash_w-1, dash_h-1), (255,255,255), 2) 
                
                thumb2 = cv2.resize(sliding_window_img, (dash_w, dash_h))
                cv2.rectangle(thumb2, (0,0), (dash_w-1, dash_h-1), (255,255,255), 2) 
                
                panels = [(thumb1, "1. Edge Mask"), (thumb2, "2. Sliding Window"), (radar_resized, "3. Radar (FCW)")]

                gap, margin_x, margin_y = 5, 5, 5
                
                for i, (img_panel, title) in enumerate(panels):
                    y_pos = margin_y + i * (dash_h + gap)
                    x_pos = w - dash_w - margin_x
                    
                    if y_pos + dash_h <= h and x_pos >= 0:
                        frame[y_pos:y_pos+dash_h, x_pos:x_pos+dash_w] = img_panel
                        fs = max(0.3, dash_w / 800)
                        cv2.putText(frame, title, (x_pos+5, y_pos+15), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1)
                            
        return frame, radar
