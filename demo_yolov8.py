import argparse
import os
import cv2
import json
import numpy as np
import torch
from torchvision.transforms import functional as F
from utils_vitpose.model import vitpose_base
from utils_vitpose.common import box2cs, affine_transformation
from ultralytics import YOLO


coco_connections = [
    (0, 1),  # Nose to left eye
    (0, 2),  # Nose to right eye
    (1, 3),  # Left eye to left ear
    (2, 4),  # Right eye to right ear
    (0, 5),  # Nose to left shoulder
    (0, 6),  # Nose to right shoulder
    (5, 6),  # Left shoulder to right shoulder
    (5, 7),  # Left shoulder to left elbow
    (7, 9),  # Left elbow to left wrist
    (6, 8),  # Right shoulder to right elbow
    (8, 10), # Right elbow to right wrist
    (5, 11), # Left shoulder to left hip
    (6, 12), # Right shoulder to right hip
    (11, 12),# Left hip to right hip
    (11, 13),# Left hip to left knee
    (13, 15),# Left knee to left ankle
    (12, 14),# Right hip to right knee
    (14, 16) # Right knee to right ankle
]

def convert_to_json(joints):
    json_result = [{
        'x': joint[0],
        'y': joint[1],
        'presence': joint[2]
    } for joint in joints]
    return json_result


def draw_landmarks_on_image(frame, keypoints):
    annotated_frame = frame.copy()

    for keypoint in keypoints:
        x, y, conf_score = keypoint
        cv2.circle(annotated_frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    for connection in coco_connections:
        start_idx, end_idx = connection
        start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
        end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
        cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)

    return annotated_frame


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO("weights/yolov8n-pose.pt").to(device)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    if args.save_visualization:
        annotated_frames = []
    if args.save_json:
        json_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            result = model(frame)[0]
        except IndexError:
            if args.save_json:
                json_frames.append(None)
            continue
        
        xy = result.keypoints.xy
        conf = result.keypoints.conf
        if conf.shape[0] > 1: # Detected multiple persons
            highest_confidence = torch.argmax(torch.mean(conf, dim=1))
            xy = xy[highest_confidence]
            conf = conf[highest_confidence]
        else:
            xy, conf = xy.squeeze(), conf.squeeze()
        out = torch.concat([xy, conf.unsqueeze(-1)], dim=1).cpu().numpy()
        
        if args.save_json:
            json_result = convert_to_json(out)
            json_frames.append(json_result)

        if args.visualize or args.save_visualization:
            annotated_frame = draw_landmarks_on_image(frame, out)
        
        if args.visualize:
            cv2.imshow("video", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
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
        
        output_file = f"{video_name}_annotated.mp4"
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

    main(args)
