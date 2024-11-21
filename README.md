# Crowd Detection using AI  

## Overview  
This project involves training an AI model to detect persons in a video and analyzing the detected data to identify crowds. It leverages pre-trained object detection models and applies custom logic to detect and log crowd events.  

## Features  
- Detect persons in video frames using pre-trained object detection models.  
- Analyze frame data to identify crowds based on defined criteria.  
- Log crowd events in a structured CSV file.  

## Crowd Detection Logic  
A crowd is defined based on the following criteria:  
1. A crowd consists of **three or more persons** standing close together in a single frame.  
2. The group of persons must persist for at least **10 consecutive frames** to qualify as a crowd.  
3. When a crowd is detected:  
   - Log the frame number.  
   - Record the count of persons in the crowd.  

The results are saved in a CSV file with the following structure:  
| Frame Number | Person Count in Crowd |  
|--------------|-----------------------|  

## Getting Started  

### Prerequisites  
- Python 3.8 or later  
- Required libraries:  
  - `OpenCV`  
  - `TensorFlow` or `PyTorch` (for pre-trained models)  
  - `numpy`  
  - `pandas`  

### Installation  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/crowd-detection.git  
   cd crowd-detection
   pip install -r requirements.txt  

