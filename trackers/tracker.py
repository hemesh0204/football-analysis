from ultralytics import YOLO  # type: ignore
import supervision as sv  # type: ignore
import pickle
import os
from utils import get_center, get_bbox_wdth, get_foot_position
import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd


class Tracker:
    def __init__(self, model_path="models/best.pt"):
        # Load YOLO model from the provided path
        self.model = YOLO(model_path)
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate ball positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]
        return ball_positions

    def detect_frames(self, frames):
        """
        One valid question would be why to predict when we can track directly, the problem is from detection dataset,
        which we got from roboflow, which has goalkeeper, and due to small sample size, the goalkeeper is not being
        detected properly. So we will change the goalkeeper to player to make our lives easier, as it is not serving
        any purpose to identify the goalkeeper.
        """
        # detections = self.model.predict(frames) but doing this may lead to memory issue we are gonna detect in batches

        batches = 20
        detection = []

        for batch in range(0, len(frames), batches):
            # Perform detection on batches of frames with a confidence threshold of 0.1
            batch_detection = self.model.predict(
                frames[batch : batch + batches], conf=0.1
            )
            detection.extend(batch_detection)

        return detection

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_eclipse(self, frame, bbox, color, track_id=None):
        # we want the eclipse to be at the botton, so we will use y2
        y2 = bbox[3]
        x_center, _ = get_center(bbox)
        width = get_bbox_wdth(bbox)

        cv2.ellipse(
            frame,
            center=(
                int(x_center),
                int(y2),
            ),  # Ensure that the center coordinates are integers
            axes=(int(width), int(0.35 * width)),  # Ensure the axes are also integers
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ],
            dtype=np.int32,
        )

        # Because opencv expects this in this format
        triangle_points = triangle_points.reshape((-1, 1, 2))
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Convert team_ball_control list to a NumPy array to use array-based indexing
        team_ball_control_till_frame = np.array(team_ball_control[: frame_num + 1])

        # Get the number of times each team had ball control
        team_1_num_frames = (team_ball_control_till_frame == 1).sum()
        team_2_num_frames = (team_ball_control_till_frame == 2).sum()

        # Calculate the percentage of ball control for each team
        if team_1_num_frames + team_2_num_frames > 0:
            team_1_percentage = (
                team_1_num_frames / (team_1_num_frames + team_2_num_frames) * 100
            )
            team_2_percentage = (
                team_2_num_frames / (team_1_num_frames + team_2_num_frames) * 100
            )
        else:
            team_1_percentage = 50
            team_2_percentage = 50

        # Display ball control percentages
        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1_percentage:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2_percentage:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotation(self, video_frames, tracks, team_ball_control):
        """
        Draw a small ellipse beneath each player, referee, and ball.
        """
        op_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # so the original frame is not being changed

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            refree_dict = tracks["referees"][frame_num]

            # Draw for players
            for track_id, player in player_dict.items():
                # Ensure the team_color is assigned properly
                color = player.get("team_color", None)

                # Fallback to red if team_color is not set
                if color is None:
                    print(
                        f"Warning: Player {track_id} does not have a team color assigned."
                    )
                    color = (0, 0, 255)  # Red as fallback color

                frame = self.draw_eclipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw for referees
            for track_id, referee in refree_dict.items():
                frame = self.draw_eclipse(frame, referee["bbox"], (0, 255, 255))

            # Draw for ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Add the updated frame to the output video frames
            op_video_frames.append(frame)

        return op_video_frames
