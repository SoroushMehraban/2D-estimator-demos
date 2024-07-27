import argparse
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


FPS = 30
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def convert_to_json(detection_result, height, width):
    try:
        landmarks = detection_result.pose_landmarks[0]
    except IndexError:
        return None
    json_result = [{
        'x': landmark.x * width,
        'y': landmark.y * height,
        # 'z': landmark.z,
        'visibility': landmark.visibility,
        'presence': landmark.presence
    } for landmark in landmarks]
    return json_result


def main(args):
    model_path = 'weights/pose_landmarker_full.task'
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                    running_mode=VisionRunningMode.VIDEO)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 1
        if args.save_visualization:
            annotated_frames = []
        if args.save_json:
            json_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            height, width, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            frame_timestamp_ms = int(1000 / FPS * frame_idx)
            frame_idx += 1
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if args.save_json:
                json_result = convert_to_json(pose_landmarker_result, height, width)
                json_frames.append(json_result)

            if args.visualize or args.save_visualization:
                annotated_frame = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
            
            if args.visualize:
                cv2.imshow("video", annotated_frame)
                cv2.waitKey(25)
            
            if args.save_visualization:
                annotated_frames.append(annotated_frame)
    
    
    cap.release()
    cv2.destroyAllWindows()

    
    if args.save_json:
        video_name = args.video.split(os.sep)[-1].split(".")[0]
        with open(f'{video_name}.json', 'w') as fp:
            json.dump(json_frames, fp)

    if args.save_visualization:
        video_name = args.video.split(os.sep)[-1].split(".")[0]
        
        output_file = f"{video_name}_mediapipe.mp4"
        fps = 30
        size = (width, height)
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for frame in annotated_frames:
            out.write(frame)
        
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, metavar='PATH', required=True,
                        help='Path to where video is located')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save-visualization', action='store_true')
    parser.add_argument('--save-json', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(e)
