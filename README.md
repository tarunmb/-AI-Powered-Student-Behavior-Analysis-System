# -AI-Powered-Student-Behavior-Analysis-System
# Student Behavior Analysis System

An AI-powered system that uses computer vision and deep learning to automatically detect and analyze student behaviors in real-time. This system helps educators monitor classroom activities and gain insights into student engagement patterns.

## Features

- 🎥 Real-time behavior detection using YOLOv8
- 📸 Support for both image and video processing
- 🔄 Multi-threaded video processing for improved performance
- 📊 Behavior tracking and analysis
- 🌐 Web-based interface for easy access
- 📱 Responsive design for various devices

## Detected Behaviors

The system can detect the following behaviors:
- Using laptop
- Laughing
- Looking away
- Using mobile phone
- Reading
- Sleeping
- Writing

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam (for real-time processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-behaviour-analysis.git
cd student-behaviour-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
student-behaviour-analysis/
├── app.py              # Main application file
├── models/            # Directory for YOLO models
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
├── data/             # Directory for uploaded files
│   ├── images/       # Uploaded images
│   └── videos/       # Uploaded videos
└── output/           # Processed output files
    ├── images/       # Processed images
    └── videos/       # Processed videos
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5001
```

3. Upload an image or video file through the web interface

4. View the processed results and behavior analysis

## Real-time Processing

The system supports real-time video processing through your webcam:
1. Click on the "Start Real-time Analysis" button
2. Allow camera access when prompted
3. View real-time behavior detection results

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Upload and process media files
- `POST /process_frame`: Process real-time video frames
- `GET /static/<folder>/<filename>`: Access processed files

## Performance Optimization

- Multi-threaded video processing
- Frame skipping for better performance
- Optimized YOLOv8 model inference
- Efficient resource management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for object detection
- Flask for web framework
- OpenCV for image processing
- Ultralytics for YOLO implementation

## Contact

For any questions or suggestions, please open an issue in the GitHub repository.
