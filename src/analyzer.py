#!/usr/bin/env python3
"""
analyzer.py -- Inference and Feedback Overlay for Swimming Analytics

Usage:
    python analyzer.py --input INPUT_VIDEO --model_dir models/ --output OUTPUT_VIDEO --window 5 --stride 1
    OR
    python analyzer.py --input INPUT_VIDEO --model_dir models/ --output OUTPUT_VIDEO
    OR
    python analyzer.py --input INPUT_VIDEO --output OUTPUT_VIDEO 

Arguments:
    --input       Path to input video (.mp4)
    --model_dir   Directory with trained model files (default: chosen_models/)
    --output      Output video path (.mp4 or .avi)
    --window      Sliding window size (default: 5)
    --stride      Sliding window stride (default: 1)
"""

import os
import cv2
import numpy as np
import joblib
import argparse
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

# --- Feedback mapping per motion + label ---
FEEDBACK_MAP = {
    "kick": {
        "consistent_kick": "Your kick looks consistent—great job!",
        "inconsistent_kick": "Your kick is inconsistent. Try dryland flutter-kick drills to stabilize your ankle motion.",
    },
    "breathing": {
        "balanced_breathing": "Breathing appears balanced.",
        "breathing_bias_detected": "You have a breathing bias. Practice bilateral breathing drills to even it out.",
    },
    "hand_entry": {
        "smooth_hand_entry": "Smooth hand entry detected. Excellent symmetry and control!",
        "uneven_hand_entry": "Hand entry is uneven. Focus on alternating entry drills to improve stroke balance.",
        "asymmetric_hand_entry": "Asymmetric hand entry detected. Work on aligning shoulder motion through mirror drills or snorkel training.",
        "unstable_hand_entry": "Elbow angle is unstable during entry. Strengthen scapular control and triceps through resistance band drills.",
    },
    "overall_posture": {
        "stable_posture": "Posture is stable throughout the stroke.",
        "unstable_posture": "Posture is unstable.",
    }
}

MOTIONS = ["kick", "breathing", "hand_entry", "overall_posture"]

#Function to load all models from the specified directory
def load_all_models(model_dir="chosen_models"):
    models = {}
    for motion in MOTIONS:
        candidates = [f for f in os.listdir(model_dir) if f.startswith(motion + "_")]
        if not candidates:
            raise FileNotFoundError(f"No model found for motion '{motion}' in {model_dir}")
        model_path = os.path.join(model_dir, candidates[0])
        models[motion] = joblib.load(model_path)
    return models


# Function to overlay feedback text on the video frame
def overlay_all_feedback(frame, aspect_results):
    # Settings
    font = cv2.FONT_HERSHEY_COMPLEX  
    font_scale = 0.5
    text_color = (20, 20, 20)        # Very dark gray (almost black)
    bg_color = (240, 240, 240)       # Light gray background for contrast
    thickness = 1
    margin = 10
    y = 30

    for motion, result in aspect_results.items():
        text = f"{result['label']} | {result['feedback']}"

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Draw filled rectangle as background
        cv2.rectangle(
            frame,
            (margin-5, y - text_h - 5),
            (margin + text_w + 5, y + 5),
            bg_color,
            thickness=cv2.FILLED
        )
        # Draw text over rectangle
        cv2.putText(
            frame,
            text,
            (margin, y),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )
        y += text_h + 20  # Move down for next line

    return frame


# Functions to compute features from keypoints (same as used in training)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def vector_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.arccos(dot_product)

def compute_velocity(kps_window, joint_id):
    positions = [np.array(frame[joint_id][:2]) for frame in kps_window if frame]
    if len(positions) < 2:
        return 0.0
    displacements = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
    mean_disp = np.mean(displacements)
    velocity = mean_disp * 30  # Assuming 30 FPS
    return velocity

def compute_symmetry(kps_window, joint_id_left, joint_id_right):
    diffs = []
    for frame in kps_window:
        if frame:
            diff = abs(frame[joint_id_left][0] - frame[joint_id_right][0])
            diffs.append(diff)
    if not diffs:
        return 0.0
    return np.mean(diffs)

def compute_angle_variance(kps_window, joint_a, joint_b, joint_c):
    angles = []
    for frame in kps_window:
        if frame:
            a = np.array(frame[joint_a][:2])
            b = np.array(frame[joint_b][:2])
            c = np.array(frame[joint_c][:2])
            v1 = a - b
            v2 = c - b
            angle = vector_angle(v1, v2)
            angles.append(angle)
    if not angles:
        return 0.0
    return np.var(angles)

def compute_features(kps_window):
    mp_lm = mp_pose.PoseLandmark
    ankle_ids = [mp_lm.LEFT_ANKLE.value, mp_lm.RIGHT_ANKLE.value]
    y_coords = [np.mean([frame[i][1] for i in ankle_ids]) for frame in kps_window if frame]
    ankle_y_variance = np.var(y_coords) if len(y_coords) > 1 else 0.0

    left_wrist_velocity = compute_velocity(kps_window, mp_lm.LEFT_WRIST.value)
    right_wrist_velocity = compute_velocity(kps_window, mp_lm.RIGHT_WRIST.value)
    velocity_smoothness = np.abs(left_wrist_velocity - right_wrist_velocity)

    shoulder_symmetry = compute_symmetry(kps_window, mp_lm.LEFT_SHOULDER.value, mp_lm.RIGHT_SHOULDER.value)

    left_elbow_angle_var = compute_angle_variance(kps_window, mp_lm.LEFT_SHOULDER.value, mp_lm.LEFT_ELBOW.value, mp_lm.LEFT_WRIST.value)
    right_elbow_angle_var = compute_angle_variance(kps_window, mp_lm.RIGHT_SHOULDER.value, mp_lm.RIGHT_ELBOW.value, mp_lm.RIGHT_WRIST.value)
    elbow_angle_variance = (left_elbow_angle_var + right_elbow_angle_var) / 2 if (left_elbow_angle_var and right_elbow_angle_var) else 0.0

    left_hip_angle_var = compute_angle_variance(kps_window, mp_lm.LEFT_SHOULDER.value, mp_lm.LEFT_HIP.value, mp_lm.LEFT_KNEE.value)
    right_hip_angle_var = compute_angle_variance(kps_window, mp_lm.RIGHT_SHOULDER.value, mp_lm.RIGHT_HIP.value, mp_lm.RIGHT_KNEE.value)
    hip_angle_variance = (left_hip_angle_var + right_hip_angle_var) / 2 if (left_hip_angle_var and right_hip_angle_var) else 0.0

    features = {
        "ankle_y_variance": ankle_y_variance,
        "velocity_smoothness": velocity_smoothness,
        "shoulder_symmetry": shoulder_symmetry,
        "elbow_angle_variance": elbow_angle_variance,
        "hip_angle_variance": hip_angle_variance,
    }
    return features

# Function to predict all aspects using the loaded models
def predict_all_aspects(features, models, feedback_map):
    results = {}
    for motion, model in models.items():
        X = [features[f] for f in sorted(features.keys())]
        pred = model.predict([X])[0]
        feedback = feedback_map[motion].get(pred, "No feedback mapping")
        results[motion] = {"label": pred, "feedback": feedback}
    return results

# Main function to handle video processing and inference
def main():
    parser = argparse.ArgumentParser(description="Swimming Analytics Inference and Feedback Overlay")
    parser.add_argument("--input", required=True, help="Input video file (.mp4)")
    parser.add_argument("--model_dir", default="models", help="Directory with trained models")
    parser.add_argument("--output", required=True, help="Output video file (.mp4 or .avi)")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")

    args = parser.parse_args()

    # Load all models
    models = load_all_models(args.model_dir)

    # Video setup
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or fps is None:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_dim = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, resize_dim)

    keypoint_buffer = []
    last_aspect_results = {motion: {"label": "waiting", "feedback": "Analyzing..."} for motion in MOTIONS}

    frame_count = 0

    while cap.isOpened() :
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, resize_dim)
        results = pose.process(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_resized,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            keypoint_buffer.append(landmarks)
        else:
            keypoint_buffer.append(np.zeros((33, 3)).tolist())

        # Only predict when buffer is full
        if len(keypoint_buffer) >= args.window:
            kps_window = keypoint_buffer[-args.window:]
            features = compute_features(kps_window)
            aspect_results = predict_all_aspects(features, models, FEEDBACK_MAP)
            last_aspect_results = aspect_results

        # Overlay latest available predictions for all aspects
        frame_annotated = overlay_all_feedback(frame_resized, last_aspect_results)
        out.write(frame_annotated)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[✓] Processed {frame_count} frames. Output saved to {args.output}")

if __name__ == "__main__":
    main()
