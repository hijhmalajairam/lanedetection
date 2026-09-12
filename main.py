import cv2
import os
import sys
import urllib.request
from src.pipeline import LDWSPipeline

def download_haar_cascade():
    url = "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml"
    if not os.path.exists("cars.xml"):
        print("Downloading Vehicle Detection Model (cars.xml)...")
        urllib.request.urlretrieve(url, "cars.xml")
        print("Download complete.")

def process_video(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Could not find video at {input_path}")
        sys.exit(1)
        
    download_haar_cascade()
        
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(os.path.dirname(input_path), '..', 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
    
    pipeline = LDWSPipeline()
    
    print("-----------------------------------------")
    print("FULL ADAS SUITE ACTIVE")
    print(" -> LDWS (Lane Departure)")
    print(" -> LKAS (Lane Keep Assist Steering)")
    print(" -> FCW  (Forward Collision Warning)")
    print("-----------------------------------------")
    print(" Controls:")
    print(" Press 'D' -> Toggle Dashboard On/Off")
    print(" Press 'L' -> Simulate Left Turn Signal")
    print(" Press 'R' -> Simulate Right Turn Signal")
    print(" Press 'O' -> Turn off Signals")
    print(" Press 'Q' -> Quit & Save Video")
    print("-----------------------------------------")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video finished. Saved to output_video.mp4")
            break
            
        processed_frame = pipeline.process_frame(frame)
        display_frame = cv2.resize(processed_frame, (1280, 720))
        out.write(display_frame)
        cv2.imshow("Advanced Driver Assistance System", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Processing stopped. Saved to output_video.mp4")
            break
        elif key == ord('d'):
            pipeline.show_dashboard = not pipeline.show_dashboard
            print(f"Dashboard {'ON' if pipeline.show_dashboard else 'OFF'}")
        elif key == ord('l'):
            pipeline.turn_signal = 'left'
            print("Left Signal ON")
        elif key == ord('r'):
            pipeline.turn_signal = 'right'
            print("Right Signal ON")
        elif key == ord('o'):
            pipeline.turn_signal = None
            print("Turn Signals OFF")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = os.path.join("test_videos", "project_video.mp4")
    process_video(video_path)
