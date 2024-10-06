from utils import save_video, read_video
from trackers import Tracker
from teams_assigner import teamAssigner
import cv2
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_test_estimator import SpeedAndDistance_Estimator

def main():
    # Read the video
    frames = read_video('Input/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pkl')

    
    # Get object position
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    
    # View Transfomer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # speed and distanc eestimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    
    # Assign teams based on player colors
    team_assigner = teamAssigner()  # Ensure this class is working correctly
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # Assign team colors to each player in every frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Get team ID and color for each player
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)

            # Assign team color to the player
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball acquisition
    player_ball_assigner_obj = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner_obj.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        

    
    # Draw annotations (ellipses and rectangles)
    output_video_frames = tracker.draw_annotation(frames, tracks, team_ball_control)

    
    # draw camera movement estimator
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    
    # draw the speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the video
    save_video(output_video_frames, 'output_video/output_video.avi')


if __name__ == '__main__':
    main()
