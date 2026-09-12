from flask import Flask, Response, request, jsonify, redirect, send_from_directory
import cv2
import os
import threading
import urllib.request
from src.pipeline import LDWSPipeline

app = Flask(__name__)
pipeline = LDWSPipeline()

# Shared state for video source
current_video = 'test_videos/project_video.mp4'
video_lock = threading.Lock()
video_changed = threading.Event()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>ADAS Web Command Center</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #fff; margin: 0; display: flex; height: 100vh; overflow: hidden; }
        .video-container { flex: 3; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .controls { flex: 1; min-width: 300px; background: #1e1e1e; padding: 30px; overflow-y: auto; box-shadow: -5px 0 15px rgba(0,0,0,0.5); }
        #feed { max-width: 100%; max-height: 85vh; border: 4px solid #333; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); }
        h1 { margin-top: 0; color: #00e5ff; }
        h2 { font-size: 14px; border-bottom: 1px solid #444; padding-bottom: 5px; margin-top: 25px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
        .btn-group { display: flex; gap: 10px; margin-bottom: 15px; }
        button { flex: 1; background: #2d2d2d; color: #fff; border: 1px solid #444; padding: 12px; cursor: pointer; border-radius: 6px; font-weight: bold; transition: 0.2s;}
        button:hover { background: #3d3d3d; border-color: #00e5ff; }
        button:active { background: #00e5ff; color: #000; }
        .checkbox-container { display: flex; align-items: center; gap: 10px; margin-top: 15px; font-size: 15px; font-weight: bold; cursor: pointer; }
        input[type=checkbox] { width: 18px; height: 18px; accent-color: #00e5ff; cursor: pointer; }
        .info-box { background: #2a2a2a; border-left: 3px solid #00e5ff; padding: 12px 15px; margin-top: 15px; border-radius: 4px; font-size: 13px; color: #ccc; line-height: 1.6; }
        .info-box strong { color: #00e5ff; }
        .feature-list { list-style: none; padding: 0; margin: 10px 0; }
        .feature-list li { padding: 6px 0; border-bottom: 1px solid #2a2a2a; font-size: 13px; }
        .feature-list li::before { content: "\\2713  "; color: #00e5ff; font-weight: bold; }
        .status { font-size: 12px; color: #00e5ff; margin-top: 5px; min-height: 18px; }

        /* Upload Zone */
        .upload-zone { border: 2px dashed #444; border-radius: 10px; padding: 25px; text-align: center; cursor: pointer; transition: 0.3s; margin-top: 10px; }
        .upload-zone:hover { border-color: #00e5ff; background: #1a2a2a; }
        .upload-zone.dragging { border-color: #00e5ff; background: #1a3a3a; }
        .upload-zone p { margin: 5px 0; color: #888; font-size: 13px; }
        .upload-zone .icon { font-size: 36px; }
        .upload-zone input { display: none; }
        .now-playing { font-size: 12px; color: #00e5ff; margin-top: 8px; word-break: break-all; }
    </style>
</head>
<body>
    <div id="startupOverlay" style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.95);z-index:9999;display:flex;align-items:center;justify-content:center;color:#00e5ff;font-family:sans-serif;cursor:pointer;flex-direction:column;" onclick="startADAS()">
        <h1 style="font-size:3em; margin-bottom:10px;">ADAS System Initialization</h1>
        <p style="font-size:1.5em;color:#fff;">Click anywhere to enable audio and start.</p>
    </div>

    <div class="video-container">
        <h2 style="color: #666; border: none; margin: 0 0 10px 0;">Live ADAS Camera Feed</h2>
        <img id="feed" src="">
    </div>
    
    <div class="controls">
        <h1>Control Panel</h1>

        <h2>Upload Video</h2>
        <div class="upload-zone" id="dropzone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">&#128249;</div>
            <p><strong>Click or Drag & Drop</strong></p>
            <p>Upload any dashcam video (.mp4, .avi, .mov)</p>
            <input type="file" id="fileInput" accept="video/*">
        </div>
        <button style="width:100%; margin-top:10px; background:#0055ff; color:#fff; border:none; padding:10px; border-radius:4px; font-weight:bold; cursor:pointer;" onclick="switchVideo('test_videos/project_video.mp4', 'Main Video')">&#10227; Reset to Main Video (Completed)</button>
        <div class="now-playing" id="nowPlaying">Now playing: project_video.mp4</div>
        <div class="status" id="status"></div>

        <h2 style="color: #ffaa00; margin-top: 25px;">🚧 Experimental (WIP)</h2>
        <p style="font-size: 11px; color: #aaa; margin-top: -10px; margin-bottom: 10px;">Calibration models for these test videos are still under development.</p>
        <style>
            .video-list { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 15px; }
            .vid-btn { background: #222; border: 1px solid #444; color: #ccc; padding: 6px; border-radius: 4px; font-size: 11px; cursor: pointer; transition: 0.2s; flex: 1 1 45%; }
            .vid-btn:hover { background: #ffaa00; color: #000; border-color: #ffaa00; font-weight: bold; }
        </style>
        <div class="video-list">
            <button class="vid-btn" onclick="switchVideo('C:/Users/Asus/Downloads/videos/13588904_3840_2160_30fps.mp4', 'WIP: Dashcam 1')">Test Video 1</button>
            <button class="vid-btn" onclick="switchVideo('C:/Users/Asus/Downloads/videos/13858526-hd_1920_1080_50fps.mp4', 'WIP: Dashcam 2')">Test Video 2</button>
            <button class="vid-btn" onclick="switchVideo('C:/Users/Asus/Downloads/videos/16094915_1920_1080_50fps.mp4', 'WIP: Dashcam 3')">Test Video 3</button>
            <button class="vid-btn" onclick="switchVideo('C:/Users/Asus/Downloads/videos/5382495-uhd_3840_2160_24fps.mp4', 'WIP: Dashcam 4')">Test Video 4</button>
            <button class="vid-btn" onclick="switchVideo('C:/Users/Asus/Downloads/videos/855980-hd_1920_1080_30fps.mp4', 'WIP: Dashcam 5')">Test Video 5</button>
        </div>

        <h2>Camera Calibration</h2>
        <label class="checkbox-container" style="font-size: 13px; margin-top: 5px; margin-bottom: 15px; color: #00e5ff;">
            <input type="checkbox" id="auto_calib" checked> 
            Auto-Calibrate (Default)
        </label>
        
        <div class="slider-container">
            <label>Horizon Line (Top Crop) <span id="top_y_val">0.63</span></label>
            <input type="range" id="top_y" min="0.4" max="0.8" step="0.01" value="0.63" disabled>
        </div>
        
        <div class="slider-container">
            <label>Dashboard Mask (Bottom Crop) <span id="bottom_y_val">0.95</span></label>
            <input type="range" id="bottom_y" min="0.6" max="1.0" step="0.01" value="0.95" disabled>
        </div>
        
        <div class="slider-container">
            <label>Perspective Width <span id="top_w_val">0.10</span></label>
            <input type="range" id="top_w" min="0.05" max="0.5" step="0.01" value="0.10" disabled>
        </div>

        <h2>UI Display Mode</h2>
        <style>
            .view-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 15px; }
            .view-card { background: #1a1a1a; border: 2px solid #333; border-radius: 6px; padding: 0; text-align: center; cursor: pointer; transition: 0.2s; overflow: hidden; }
            .view-card:hover { border-color: #00e5ff; }
            .view-card.active { border-color: #00e5ff; box-shadow: 0 0 10px rgba(0,229,255,0.5); }
            .preview { width: 100%; height: 55px; position: relative; overflow: hidden; }
            .view-label { font-size: 9px; color: #aaa; font-weight: bold; text-transform: uppercase; padding: 4px 0; background: #222; letter-spacing: 0.5px; }
            .view-card.active .view-label { color: #00e5ff; }

            /* Final HUD - road with green lane */
            .pv-final { background: linear-gradient(to bottom, #556, #444 40%, #555 60%); }
            .pv-final::after { content:''; position:absolute; bottom:0; left:20%; right:20%; height:60%; 
                background: linear-gradient(to top, rgba(0,255,0,0.4), transparent); clip-path: polygon(35% 0, 65% 0, 100% 100%, 0 100%); }

            /* ROI - trapezoid outline */
            .pv-roi { background: linear-gradient(to bottom, #334, #445 50%, #556); }
            .pv-roi::after { content:''; position:absolute; top:10%; left:10%; right:10%; bottom:10%;
                border: 2px solid #0f0; clip-path: polygon(30% 0, 70% 0, 100% 100%, 0 100%); background: rgba(0,255,0,0.15); }

            /* Edge Mask - black and white */
            .pv-edge { background: #000; }
            .pv-edge::before { content:''; position:absolute; bottom:0; left:25%; width:3px; height:70%; background:#fff; transform:rotate(-8deg); }
            .pv-edge::after { content:''; position:absolute; bottom:0; right:25%; width:3px; height:70%; background:#fff; transform:rotate(8deg); }

            /* Hough - red lines on dark road */
            .pv-hough { background: linear-gradient(to bottom, #223, #334); }
            .pv-hough::before { content:''; position:absolute; bottom:0; left:30%; width:2px; height:80%; background:#f00; transform:rotate(-12deg); }
            .pv-hough::after { content:''; position:absolute; bottom:0; right:30%; width:2px; height:80%; background:#f00; transform:rotate(12deg); }

            /* Bird's eye - top down warped */
            .pv-warp { background: #111; }
            .pv-warp::before { content:''; position:absolute; top:0; bottom:0; left:35%; width:2px; background:#fff; }
            .pv-warp::after { content:''; position:absolute; top:0; bottom:0; right:35%; width:2px; background:#fff; }

            /* Sliding window - green boxes */
            .pv-slide { background: #111; }
            .pv-slide::before { content:''; position:absolute; left:30%; top:10%; width:12px; height:10px; border:2px solid #0f0; }
            .pv-slide::after { content:''; position:absolute; right:30%; top:30%; width:12px; height:10px; border:2px solid #0f0; }

            /* Optical flow - cyan streaks */
            .pv-flow { background: linear-gradient(to bottom, #223, #334); }
            .pv-flow::before { content:''; position:absolute; top:20%; left:20%; width:40px; height:2px; background:#0ff; transform:rotate(30deg); }
            .pv-flow::after { content:''; position:absolute; top:50%; right:20%; width:30px; height:2px; background:#0ff; transform:rotate(-20deg); }

            /* Radar - dark with green circle */
            .preview-radar { background: #111; }
            .preview-radar::before { content:''; position:absolute; top:50%; left:50%; width:30px; height:30px; border:1px solid #0f0; border-radius:50%; transform:translate(-50%,-50%); }
            .preview-radar::after { content:''; position:absolute; bottom:15%; left:50%; width:6px; height:10px; background:#fff; transform:translateX(-50%); }

            /* Clean view */
            .preview-clean { background: #222; }
        </style>
        
        <div class="view-grid">
            <div class="view-card active" onclick="setView('final', this)">
                <div class="preview preview-final"></div>
                <span>Final HUD</span>
            </div>
            <div class="view-card" onclick="setView('roi', this)">
                <div class="preview preview-roi"></div>
                <span>ROI Crop</span>
            </div>
            <div class="view-card" onclick="setView('edge', this)">
                <div class="preview preview-edge"></div>
                <span>Edge Mask</span>
            </div>
            <div class="view-card" onclick="setView('hough', this)">
                <div class="preview preview-hough"></div>
                <span>Hough Lines</span>
            </div>
            <div class="view-card" onclick="setView('warp', this)">
                <div class="preview preview-warp"></div>
                <span>Bird's-Eye</span>
            </div>
            <div class="view-card" onclick="setView('sliding', this)">
                <div class="preview preview-sliding"></div>
                <span>Polynomials</span>
            </div>
            <div class="view-card" onclick="setView('flow', this)">
                <div class="preview preview-flow"></div>
                <span>Optical Flow</span>
            </div>
            <div class="view-card" onclick="setView('radar', this)">
                <div class="preview preview-radar"></div>
                <span>Radar Map</span>
            </div>
            <div class="view-card" onclick="setView('clean', this)">
                <div class="preview preview-clean"></div>
                <span>Clean View</span>
            </div>
        </div>
        
        <div style="margin-top:20px;">
            <button class="turn-btn" onclick="sendSettings({turn_signal: 'left'})">&#8592; L Turn</button>
            <button class="turn-btn" onclick="sendSettings({turn_signal: null})">Off</button>
            <button class="turn-btn" onclick="sendSettings({turn_signal: 'right'})">R Turn &#8594;</button>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const statusEl = document.getElementById('status');
        const nowPlaying = document.getElementById('nowPlaying');
        
        function syncSliders(preset) {
            if(preset) {
                document.getElementById('top_y').value = preset.top_y;
                document.getElementById('bottom_y').value = preset.bottom_y;
                document.getElementById('top_w').value = preset.top_w;
                document.getElementById('top_y_val').innerText = preset.top_y.toFixed(2);
                document.getElementById('bottom_y_val').innerText = preset.bottom_y.toFixed(2);
                document.getElementById('top_w_val').innerText = preset.top_w.toFixed(2);
            }
        }

        // Drag and Drop
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragging'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragging');
            if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
        });

        function uploadFile(file) {
            statusEl.innerText = 'Uploading ' + file.name + '...';
            const formData = new FormData();
            formData.append('video', file);

            fetch('/upload', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    statusEl.innerText = '';
                    nowPlaying.innerText = 'Now playing: ' + file.name;
                    syncSliders(data.preset);
                } else {
                    statusEl.innerText = 'Error: ' + data.error;
                }
            })
            .catch(err => { statusEl.innerText = 'Upload failed: ' + err; });
        }

        let debounceTimer = null;
        function sendSettings(data) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                fetch('/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                }).then(() => {
                    document.getElementById('status').innerText = 'Applied';
                    setTimeout(() => document.getElementById('status').innerText = '', 1500);
                });
            }, 100);
        }

        function switchVideo(path, name) {
            statusEl.innerText = 'Switching to ' + name + '...';
            fetch('/switch_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path})
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    nowPlaying.innerText = 'Now playing: ' + name;
                    statusEl.innerText = '';
                    syncSliders(data.preset);
                } else {
                    statusEl.innerText = 'Error: ' + data.error;
                }
            }).catch(err => { statusEl.innerText = 'Switch failed: ' + err; });
        }

        // Web Audio API for Warnings
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        let audioCtx = null;
        let lastBeepTime = 0;
        
        function startADAS() {
            document.getElementById('startupOverlay').style.display = 'none';
            document.getElementById('feed').src = '/video_feed';
            
            if (!audioCtx) {
                audioCtx = new AudioContext();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }
        
        function playBeep(frequency, duration) {
            if (!audioCtx || audioCtx.state === 'suspended') return; // Wait for user click
            
            const now = Date.now();
            if (now - lastBeepTime < 1000) return; // Cooldown
            lastBeepTime = now;
            
            const oscillator = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            
            oscillator.type = 'square';
            oscillator.frequency.setValueAtTime(frequency, audioCtx.currentTime);
            
            gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
            
            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            
            oscillator.start();
            oscillator.stop(audioCtx.currentTime + duration);
        }

        setInterval(() => {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.fcw) playBeep(1000, 0.4);
                    else if (data.ldws) playBeep(750, 0.3);
                }).catch(e => {});
        }, 300);

        // View Mode Grid
        function setView(mode, el) {
            document.querySelectorAll('.view-card').forEach(c => c.classList.remove('active'));
            if (el) el.classList.add('active');
            sendSettings({view_mode: mode});
        }
        
        document.getElementById('auto_calib').addEventListener('change', function() {
            const isAuto = this.checked;
            document.getElementById('top_y').disabled = isAuto;
            document.getElementById('bottom_y').disabled = isAuto;
            document.getElementById('top_w').disabled = isAuto;
            sendSettings({auto_calib: isAuto});
        });

        // Connect Sliders
        ['top_y', 'bottom_y', 'top_w'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', function() {
                    document.getElementById(id + '_val').innerText = this.value;
                    let payload = {};
                    payload[id] = parseFloat(this.value);
                    sendSettings(payload);
                });
            }
        });

        document.getElementById('dashboard').addEventListener('change', function() {
            sendSettings({show_dashboard: this.checked});
        });

        function setSignal(dir) {
            sendSettings({turn_signal: dir});
        }
    </script>
</body>
</html>
"""

def setup_models():
    models = {
        "deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        "mobilenet_iter_73000.caffemodel": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
        "stop_data.xml": "https://raw.githubusercontent.com/avivbrook/stop-sign-detection/master/stop_data.xml"
    }
    for file, url in models.items():
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(url, file)
            except Exception as e:
                print(f"Warning: Could not download {file}: {e}")

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/dl_video/<path:filename>')
def dl_video(filename):
    return send_from_directory(r'C:\Users\Asus\Downloads\videos', filename)

@app.route('/test_videos/<path:filename>')
def test_video(filename):
    return send_from_directory('test_videos', filename)

VIDEO_PRESETS = {
    'project_video.mp4': {'top_y': 0.63, 'bottom_y': 0.95, 'top_w': 0.10},
    '13588904_3840_2160_30fps.mp4': {'top_y': 0.45, 'bottom_y': 0.85, 'top_w': 0.05},
    '13858526-hd_1920_1080_50fps.mp4': {'top_y': 0.55, 'bottom_y': 0.90, 'top_w': 0.15},
    '16094915_1920_1080_50fps.mp4': {'top_y': 0.55, 'bottom_y': 0.90, 'top_w': 0.15},
    '5382495-uhd_3840_2160_24fps.mp4': {'top_y': 0.45, 'bottom_y': 0.85, 'top_w': 0.05},
    '855980-hd_1920_1080_30fps.mp4': {'top_y': 0.55, 'bottom_y': 0.90, 'top_w': 0.15},
    'default': {'top_y': 0.63, 'bottom_y': 0.95, 'top_w': 0.10}
}

def apply_video_preset(filename):
    preset = VIDEO_PRESETS.get(filename, VIDEO_PRESETS['default'])
    pipeline.top_y = preset['top_y']
    pipeline.bottom_y = preset['bottom_y']
    pipeline.top_w = preset['top_w']
    return preset

def gen_frames():
    global current_video
    setup_models()
    try:
        if pipeline.net is None:
            pipeline.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
        if pipeline.sign_cascade is None:
            pipeline.sign_cascade = cv2.CascadeClassifier('stop_data.xml')
    except Exception as e:
        print(f"Model load error: {e}")

    cap = cv2.VideoCapture(current_video)
    
    while True:
        # Check if video source changed
        if video_changed.is_set():
            video_changed.clear()
            cap.release()
            # Reset pipeline state for new video
            pipeline.left_line.reset()
            pipeline.right_line.reset()
            pipeline.M = None
            with video_lock:
                cap = cv2.VideoCapture(current_video)
        
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
            continue
        
        try:
            processed = pipeline.process_frame(frame)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Frame error (recovering): {e}")
            processed = frame
            
        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    global current_video
    if 'video' not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded"})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "error": "No file selected"})
    
    # Save to uploads folder
    save_path = os.path.join('uploads', file.filename)
    file.save(save_path)
    print(f"Uploaded: {file.filename} -> {save_path}")
    
    # Switch video source
    with video_lock:
        current_video = save_path
        pipeline.left_line.reset()
        pipeline.right_line.reset()
        preset = apply_video_preset(file.filename)
    video_changed.set()
    
    return jsonify({"status": "success", "filename": file.filename, "preset": preset})

@app.route('/switch_video', methods=['POST'])
def switch_video():
    global current_video
    data = request.get_json()
    path = data.get('path', '')
    if os.path.exists(path):
        filename = os.path.basename(path)
        with video_lock:
            current_video = path
            pipeline.left_line.reset()
            pipeline.right_line.reset()
            preset = apply_video_preset(filename)
        video_changed.set()
        return jsonify({"status": "success", "filename": filename, "preset": preset})
    return jsonify({"status": "error", "error": "File not found"}), 404

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "fcw": getattr(pipeline, 'fcw_active', False),
        "ldws": getattr(pipeline, 'ldws_active', False)
    })

@app.route('/settings', methods=['POST'])
def settings():
    data = request.json
    
    if 'auto_calib' in data:
        pipeline.auto_calib = data['auto_calib']
        if pipeline.auto_calib:
            pipeline.top_y, pipeline.bottom_y, pipeline.top_w = 0.63, 0.95, 0.10
    
    pipeline.top_y = float(data.get('top_y', pipeline.top_y))
    pipeline.bottom_y = float(data.get('bottom_y', pipeline.bottom_y))
    pipeline.top_w = float(data.get('top_w', pipeline.top_w))
    
    pipeline.turn_signal = data.get('turn_signal', pipeline.turn_signal)
    
    if 'view_mode' in data:
        pipeline.view_mode = data['view_mode']
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    print("=========================================")
    print("ADAS Web Portal Running!")
    print("Open your browser to: http://localhost:5000")
    print("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
