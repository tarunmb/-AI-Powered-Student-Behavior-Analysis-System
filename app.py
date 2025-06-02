from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import torch
from datetime import datetime
import concurrent.futures
import queue
import threading

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask app with explicit static folder path
app = Flask(__name__,
            static_url_path='',  # Empty string to serve from root
            static_folder='static')     # Directory containing static files

# Enable debug mode for development
app.config['DEBUG'] = True

# Print debug information
print(f"Base Directory: {BASE_DIR}")
print(f"Static Folder: {app.static_folder}")
print(f"Static URL Path: {app.static_url_path}")

# Define file paths using absolute paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'images')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')

# Create all necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'videos'), exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'js'), exist_ok=True)

# Load the trained YOLO model
model = YOLO('models/best.pt')  # Path to your trained YOLOv8 model

# Behavior categories (ensure these match the model's class labels)
# BEHAVIORS = ['Using_phone', 'bend', 'book', 'bow_head', 'hand-raising', 'phone', 
            #  'raise_head', 'reading', 'sleep', 'turn_head', 'upright', 'writing']

BEHAVIORS = ['laptop', 'laughing', 'looking away', 'mobile phone', 'reading', 'sleeping', 'using laptop', 'writing']

# Route for handling form submission
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process image/video
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        # Create a safe filename
        filename = file.filename.replace(' ', '_')
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the uploaded file
        file.save(file_path)

        # Check if the file is an image or a video
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            output_video_path = os.path.join(OUTPUT_FOLDER, 'videos', filename)
            # Get behavior counts from video processing
            behaviors = process_video(file_path, output_video_path)
            output_video_url = url_for('static_file', folder='videos', filename=filename)
            return render_template('index.html', 
                                video_url=output_video_url, 
                                download_url=output_video_url, 
                                filename=filename,
                                behaviors=behaviors)  # Pass behaviors to template
        
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            output_image_path = os.path.join(OUTPUT_FOLDER, 'images', filename)
            behaviors, output_image_url = process_image(file_path, output_image_path)
            return render_template('index.html', 
                                image_url=output_image_url, 
                                download_url=output_image_url, 
                                filename=filename, 
                                behaviors=behaviors)
        
        else:
            return 'Unsupported file type. Please upload an image or video file.', 400

    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return f'Error processing file: {str(e)}', 500

# Process image with YOLO and count behaviors
def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    results = model(img)  # Perform inference

    # Get class names and their corresponding counts
    behavior_counts = {behavior: 0 for behavior in BEHAVIORS}

    # For each detection, check the class and update the count
    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0].item())
            if 0 <= class_id < len(BEHAVIORS):
                behavior = BEHAVIORS[class_id]
                behavior_counts[behavior] += 1

    # Plot the bounding boxes on the image
    output_image = results[0].plot()
    cv2.imwrite(output_path, output_image)

    # Return behavior counts and output image URL
    output_image_url = url_for('static_file', folder='images', filename=os.path.basename(output_path))
    return behavior_counts, output_image_url

# Process video with YOLO
def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize behavior counters
    behavior_counts = {behavior: 0 for behavior in BEHAVIORS}
    processed_frames = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with YOLO
            results = model(frame)
            
            # Update behavior counts
            for result in results:
                for detection in result.boxes:
                    class_id = int(detection.cls[0].item())
                    if 0 <= class_id < len(BEHAVIORS):
                        behavior = BEHAVIORS[class_id]
                        behavior_counts[behavior] += 1
            
            # Draw detections on frame
            frame_output = results[0].plot()
            out.write(frame_output)
            processed_frames += 1
            
    finally:
        # Release resources
        cap.release()
        out.release()
    
    # Calculate average behaviors across all frames
    if processed_frames > 0:
        for behavior in behavior_counts:
            # Calculate the average and scale it back to total frames
            avg_count = behavior_counts[behavior] / processed_frames
            behavior_counts[behavior] = int(avg_count * total_frames)
    
    return behavior_counts

# Serve the processed static files
@app.route('/static/<folder>/<filename>')
def static_file(folder, filename):
    directory = os.path.join(OUTPUT_FOLDER, folder)
    return send_from_directory(directory, filename, as_attachment=True)

# Process real-time video frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the frame data from the request
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]  # Remove the data URL prefix
        
        # Convert base64 to image
        frame_bytes = base64.b64decode(frame_data)
        frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        
        # Process the frame with YOLO
        results = model(frame)
        
        # Get behavior counts
        behavior_counts = {behavior: 0 for behavior in BEHAVIORS}
        for result in results:
            for detection in result.boxes:
                class_id = int(detection.cls[0].item())
                if 0 <= class_id < len(BEHAVIORS):
                    behavior = BEHAVIORS[class_id]
                    behavior_counts[behavior] += 1
        
        # Get the processed frame with visualizations
        processed_frame = results[0].plot()
        
        # Convert the processed frame to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'behaviors': behavior_counts,
            'frame_data': processed_frame_data
        })
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_frame_yolo(frame):
    """Process a single frame with YOLO model"""
    # Convert frame to RGB if needed
    if len(frame.shape) == 2:  # If grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Inference
    results = model(frame)
    
    # Get detections
    pred = results.pred[0]
    behaviors = {}
    
    if len(pred):
        for *box, conf, cls in pred:
            behavior = results.names[int(cls)]
            behaviors[behavior] = behaviors.get(behavior, 0) + 1
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{behavior} {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, behaviors

def process_video_optimized(video_path, output_path):
    """Process video with optimized performance"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize behavior counters
    total_behaviors = {}
    processed_frames = 0
    
    # Create a queue for frames and results
    frame_queue = queue.Queue(maxsize=30)  # Limit queue size
    result_queue = queue.Queue()
    
    def process_frame_worker():
        while True:
            frame_data = frame_queue.get()
            if frame_data is None:  # Sentinel value to stop the thread
                break
            frame_number, frame = frame_data
            processed_frame, behaviors = process_frame_yolo(frame)
            result_queue.put((frame_number, processed_frame, behaviors))
            frame_queue.task_done()
    
    # Start worker threads
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 threads
    workers = []
    for _ in range(num_workers):
        worker = threading.Thread(target=process_frame_worker)
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    frame_number = 0
    results_buffer = {}
    next_frame_to_write = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if processing is falling behind
            if frame_number % 2 != 0:  # Process every other frame
                frame_number += 1
                continue
            
            # Add frame to queue
            frame_queue.put((frame_number, frame))
            
            # Get processed results
            while not result_queue.empty():
                idx, processed_frame, behaviors = result_queue.get()
                results_buffer[idx] = (processed_frame, behaviors)
                
                # Write frames in order
                while next_frame_to_write in results_buffer:
                    frame_to_write, frame_behaviors = results_buffer.pop(next_frame_to_write)
                    out.write(frame_to_write)
                    # Update total behaviors
                    for behavior, count in frame_behaviors.items():
                        total_behaviors[behavior] = total_behaviors.get(behavior, 0) + count
                    next_frame_to_write += 2
                    processed_frames += 1
            
            frame_number += 1
    
    finally:
        # Signal workers to stop
        for _ in workers:
            frame_queue.put(None)
        
        # Wait for workers to finish
        for worker in workers:
            worker.join()
        
        # Process remaining results
        while not result_queue.empty():
            idx, processed_frame, behaviors = result_queue.get()
            results_buffer[idx] = (processed_frame, behaviors)
        
        # Write remaining frames in order
        for idx in sorted(results_buffer.keys()):
            frame_to_write, frame_behaviors = results_buffer[idx]
            out.write(frame_to_write)
            for behavior, count in frame_behaviors.items():
                total_behaviors[behavior] = total_behaviors.get(behavior, 0) + count
            processed_frames += 1
        
        # Release resources
        cap.release()
        out.release()
    
    # Calculate average behaviors per processed frame
    for behavior in total_behaviors:
        total_behaviors[behavior] = int(total_behaviors[behavior] / processed_frames * total_frames)
    
    return total_behaviors

# Serve static files directly
@app.route('/static/js/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)

# Serve realtime.js directly
@app.route('/static/js/realtime.js')
def serve_realtime_js():
    return send_from_directory(
        os.path.join(app.static_folder, 'js'),
        'realtime.js',
        mimetype='application/javascript'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Enable debug mode for better error messages
