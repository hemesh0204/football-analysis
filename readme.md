# Football Analysis System using YOLO, OpenCV, and Machine Learning

This project demonstrates a football analysis system that uses state-of-the-art machine learning, computer vision, and deep learning techniques. The system employs **YOLO** (You Only Look Once), a cutting-edge object detector, to identify and track players, referees, and the football in real-time video footage. The project includes the integration of object tracking, team assignment using **KMeans clustering**, ball possession assignment, and the calculation of player speed and distance using **perspective transformation**.

![alt text]()

## Project Overview

In this football analysis system:

1. **YOLOv8** is used for real-time object detection of players, referees, and the ball.
2. Players are assigned to teams using **KMeans clustering**, based on the color of their jerseys.
3. A custom **optical flow** algorithm is used to measure camera movement, ensuring accurate player tracking.
4. **Perspective transformation** is applied to calculate player movement in real-world units (meters) rather than pixels.
5. **Player speed and distance** covered are calculated based on tracking data.
6. The system assigns ball possession to players and tracks which team controls the ball over time.

This project is aimed at providing a comprehensive solution to analyze football matches and can be adapted to other sports analysis contexts.

## Key Features

### 1. Object Detection with YOLOv8

- **YOLOv8** (You Only Look Once) is used to detect objects such as players, referees, and the ball in each video frame.
- The system is designed to handle real-time processing of video data for accurate analysis.

### 2. Player Team Assignment with KMeans Clustering

- Players are assigned to teams based on the dominant colors of their jerseys using **KMeans clustering**.
- Each player's jersey color is extracted from the frame and grouped using unsupervised learning to determine team assignments.

### 3. Optical Flow for Camera Movement Detection

- Optical flow is used to measure camera movement between frames. This helps to accurately track player movements even when the camera is panning or zooming.
- This ensures that player speed and distance calculations are robust to camera motion.

### 4. Perspective Transformation

- Perspective transformation is applied to the video frames to represent the depth and perspective of the field. This allows the system to convert pixel-based movement into real-world units (e.g., meters).
- This feature enables the measurement of player movements in real-world terms rather than just screen pixels.

### 5. Player Speed and Distance Calculation

- By combining the tracking data with the perspective transformation, the system can calculate the **speed** and **distance covered** by each player.
- This feature provides valuable insights into player performance during a match.

### 6. Ball Possession Assignment

- The system tracks which player is in possession of the ball by calculating the proximity of the ball to each player in every frame.
- The ball possession is assigned to the nearest player and stored as part of the tracking data.

### 7. Team Ball Control Statistics

- The system calculates the percentage of time each team has possession of the ball and displays these statistics in the video.
- This provides a comprehensive view of team performance in terms of ball control.

## Technologies Used

- **YOLOv8**: Object detection framework from the Ultralytics library.
- **OpenCV**: Image and video processing library used for drawing annotations and performing perspective transformations.
- **KMeans Clustering**: Used for unsupervised learning and grouping players into teams based on jersey color.
- **Optical Flow**: Used to measure camera movement between frames.
- **Numpy**: For handling and manipulating image data and pixel calculations.
- **Supervision**: A library for object tracking across frames using ByteTrack.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/football-analysis.git
   cd football-analysis
   ```
2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### `requirements.txt`

```txt
opencv-python==4.10.0
ultralytics==8.2.102
supervision==0.2.0
scikit-learn==1.2.2
numpy==1.24.3
```

## Usage

1. Place your input video in the `Input/` directory.
2. Ensure you have your custom-trained YOLOv8 model stored in the `models/` directory as `best.pt`.
3. Run the main script:

   ```bash
   python main.py
   ```
4. The output video, complete with object detection, player tracking, team assignment, and ball control statistics, will be saved to the `output_video/` directory.

## Project Workflow

### 1. **Object Detection**

   The system uses YOLOv8 to detect objects in each frame, including players, referees, and the football.

### 2. **Tracking and Team Assignment**

   Each player is tracked across frames using a tracking algorithm. KMeans clustering is applied to identify team memberships based on the color of each player's jersey.

### 3. **Ball Possession and Control**

   The system assigns the ball to the nearest player in each frame and keeps track of which team controls the ball throughout the match.

### 4. **Speed and Distance Calculation**

   The system calculates the speed and distance covered by each player using perspective transformation and optical flow.

### 5. **Final Video Output**

   The system generates an output video that includes:

- Annotations for player, referee, and ball positions.
- Team assignment based on jersey color.
- Ball possession statistics.
- Speed and distance metrics for each player.

## Example Use Case

- **Football Match Analysis**: The system can be used to analyze football matches, providing insights into player movements, team ball control, and overall game strategy.
- **Training Sessions**: Coaches can use the system to analyze player performance during training sessions, helping to improve player fitness and tactical understanding.

## Conclusion

This project showcases the integration of advanced machine learning and computer vision techniques to analyze football matches. By utilizing YOLO for object detection, KMeans for team assignment, and OpenCV for video processing, this system provides comprehensive insights into player and team performance. The project can be expanded to handle more complex scenarios and provide even more detailed analysis, making it an excellent tool for sports analytics.

---
