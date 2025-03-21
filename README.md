```markdown
# Accident Detection System

## Overview
The **Road Safety Website** is an interactive platform designed to enhance road safety by monitoring traffic conditions and detecting accidents in real-time. It leverages machine learning models and computer vision techniques to analyze live video feeds, providing timely alerts and data to improve traffic management and emergency response.

## Features
- Real-time accident detection from video feeds
- Web interface for monitoring and analysis
- Video upload and processing capabilities
- Customizable detection parameters
- Support for multiple video formats

## Tech Stack
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, YOLO (YOLOv4, YOLOv8-OBB)
- **Machine Learning**: Pre-trained models for traffic analysis and custom-trained CNN model for accident detection
- **Frontend**: HTML, CSS, JavaScript

## Installation
### Prerequisites
Ensure you have the following installed:
- Python (preferably in a conda environment)
- Virtual environment for Flask

### Steps to Run
#### Backend:
1. Clone the repository:
   ```sh
   git clone https://github.com/Ahhsan/roadsafety_website.git
   cd roadsafety_website/backend
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Flask server:
   ```sh
   python app.py
   ```

#### Frontend:
1. Navigate to the frontend directory:
   ```sh
   cd ../frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```

## Usage
1. Upload a video feed for analysis.
2. The system will detect and display traffic conditions, accidents, and ambulances.
3. Traffic signals will be adjusted dynamically based on real-time detection.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

## Requirements
See `requirements.txt` for a complete list of Python dependencies.

## License
[Add license information here]

## Contributors
[Add contributor information here]

## Contact
For queries, reach out via [GitHub Issues](https://github.com/Ahhsan/roadsafety_website/issues).
```

