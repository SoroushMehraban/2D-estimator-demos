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

def detect_person(model, image):
    PERSON_ID = 0

    result = model(image)[0]
    person_bboxes = torch.argwhere((result.boxes.cls == PERSON_ID)).squeeze()

    if person_bboxes.dim() == 0:  # Single person
        idx = person_bboxes.item()
    elif person_bboxes.shape[0] > 1:
        highest_score = torch.argmax(result.boxes.conf[person_bboxes])
        idx = person_bboxes[highest_score].item()
    else:
        return None

    x1, y1, x2, y2 = result.boxes.xyxy[idx].int().cpu().numpy()
    conf = result.boxes.conf[idx].int().cpu().numpy()

    w = x2 - x1
    h = y2 - y1

    return x1, y1, w, h, conf


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo = YOLO("weights/yolov8n.pt")

    model_path = 'weights/vitpose-b.pth'
    model_ckpt = torch.load(model_path)['state_dict']
    model = vitpose_base().to(device)
    model.load_state_dict(model_ckpt, strict=True)
    model.eval()
    
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

        bbox = detect_person(yolo, frame)
        if bbox is None:
            if args.save_json:
                json_frames.append(None)
            continue

        center, scale = box2cs((192, 256), bbox[:4])
        img_metas = [{
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox_score': 1,
            'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                        [13, 14], [15, 16]]  # Pair of joints in MS-COCO that are affected by mirroring
        }]

        frame_cropped = affine_transformation(frame, bbox[:4])
        
        x = F.to_tensor(frame_cropped)
        x = F.normalize(x, mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
        x = x.unsqueeze(0)

        out = model(x, img_metas)['preds'][0]
        
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
