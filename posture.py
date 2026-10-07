"""MediaPipe pose detection helpers and angle engine."""

import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from config import (
    CONNECTION_COLOR,
    CONNECTION_THICKNESS,
    DEBUG_BG_COLOR,
    DEBUG_TEXT_COLOR,
    DRAW_CONNECTIONS,
    DRAW_DEBUG_INFO,
    DRAW_HELPERS,
    DRAW_LANDMARKS,
    HELPER_LINE_COLOR,
    HELPER_LINE_THICKNESS,
    IMPORTANT_LANDMARK_COLOR,
    IMPORTANT_LANDMARK_RADIUS,
    LANDMARK_COLOR,
    LANDMARK_RADIUS,
    MAX_HEAD_TILT,
    MAX_LEAN_OFFSET,
    MAX_NECK_ANGLE,
    MAX_SHOULDER_TILT,
    MAX_SPINE_INCLINATION,
    MIN_CONFIDENCE,
    ALERT_DURATION_SECONDS,
    BREAK_REMINDER_INTERVAL_SECONDS,
    CALIBRATION_DURATION_SECONDS,
    CALIBRATION_SAMPLE_COUNT,
    CSV_FILENAME_FORMAT,
    DASHBOARD_BG_COLOR,
    DASHBOARD_BORDER_COLOR,
    DASHBOARD_INFO_COLOR,
    DASHBOARD_MUTED_COLOR,
    DASHBOARD_TEXT_COLOR,
    ENABLE_ALERTS,
    ENABLE_SCREENSHOTS,
    REPORT_DIRECTORY,
    REPORT_FILENAME_FORMAT,
    SCORE_SMOOTHING_FACTOR,
    SCREENSHOT_DIRECTORY,
    STATUS_ALERT_COLOR,
    STATUS_GOOD_COLOR,
    STATUS_HYSTERESIS_FRAMES,
    STATUS_NEUTRAL_COLOR,
    STATUS_PANEL_BG_COLOR,
    STATUS_WARNING_COLOR,
    METRICS_UPDATE_SECONDS,
)

MODEL_PATH = Path(__file__).resolve().parent / "data" / "models" / "pose_landmarker_lite.task"
LANDMARK_INDEX_MAP = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_elbow": 13,
    "right_elbow": 14,
}
IMPORTANT_LANDMARKS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
]
POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def initialize_pose_detector():
    """Create a pose detector optimized for live webcam input."""
    options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _to_pixel_point(landmark, frame_shape):
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])


def extract_landmarks(pose_landmarks, frame_shape):
    """Extract important landmarks into a dictionary of pixel coordinates."""
    landmark_map = {}
    if not pose_landmarks:
        return landmark_map

    for name, index in LANDMARK_INDEX_MAP.items():
        if index >= len(pose_landmarks):
            continue
        landmark = pose_landmarks[index]
        confidence = getattr(landmark, "visibility", 1.0)
        if confidence < MIN_CONFIDENCE:
            continue
        landmark_map[name] = _to_pixel_point(landmark, frame_shape)

    return landmark_map


def smooth_landmarks(current_landmarks, new_landmarks, alpha=0.7):
    """Blend new landmarks with the last stable frame to reduce jitter."""
    if not current_landmarks and not new_landmarks:
        return {}
    if not current_landmarks:
        return new_landmarks
    if not new_landmarks:
        return current_landmarks

    smoothed = {}
    for name in set(current_landmarks) | set(new_landmarks):
        current_point = current_landmarks.get(name)
        new_point = new_landmarks.get(name)
        if current_point is None:
            smoothed[name] = new_point
        elif new_point is None:
            smoothed[name] = current_point
        else:
            smoothed[name] = (
                int(round(current_point[0] * (1 - alpha) + new_point[0] * alpha)),
                int(round(current_point[1] * (1 - alpha) + new_point[1] * alpha)),
            )
    return smoothed


def get_landmark_coordinates(landmark_map, name):
    """Return the pixel coordinates for a named landmark if available."""
    return landmark_map.get(name)


def calculate_angle(point_a, point_b, point_c):
    """Measure the angle formed by three points in degrees."""
    if point_a is None or point_b is None or point_c is None:
        return None

    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_neck_angle(landmarks):
    """Estimate a neck angle using the nose and shoulder midpoint."""
    nose = get_landmark_coordinates(landmarks, "nose")
    left_shoulder = get_landmark_coordinates(landmarks, "left_shoulder")
    right_shoulder = get_landmark_coordinates(landmarks, "right_shoulder")

    if nose is None or left_shoulder is None or right_shoulder is None:
        return None

    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    dx = nose[0] - shoulder_midpoint[0]
    dy = nose[1] - shoulder_midpoint[1]
    return abs(math.degrees(math.atan2(dx, -dy)))


def calculate_shoulder_tilt(landmarks):
    """Estimate shoulder tilt using the line between left and right shoulders."""
    left_shoulder = get_landmark_coordinates(landmarks, "left_shoulder")
    right_shoulder = get_landmark_coordinates(landmarks, "right_shoulder")

    if left_shoulder is None or right_shoulder is None:
        return None, "N/A"

    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = math.degrees(math.atan2(dy, dx))
    direction = "left_higher" if left_shoulder[1] < right_shoulder[1] else "right_higher" if left_shoulder[1] > right_shoulder[1] else "balanced"
    return abs(angle), direction


def calculate_spine_angle(landmarks):
    """Estimate body inclination using shoulder and hip midpoints."""
    left_shoulder = get_landmark_coordinates(landmarks, "left_shoulder")
    right_shoulder = get_landmark_coordinates(landmarks, "right_shoulder")
    left_hip = get_landmark_coordinates(landmarks, "left_hip")
    right_hip = get_landmark_coordinates(landmarks, "right_hip")

    if None in (left_shoulder, right_shoulder, left_hip, right_hip):
        return None

    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_midpoint = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
    )
    dx = shoulder_midpoint[0] - hip_midpoint[0]
    dy = shoulder_midpoint[1] - hip_midpoint[1]
    return abs(math.degrees(math.atan2(dx, max(abs(dy), 1e-6))))


def calculate_head_tilt(landmarks):
    """Estimate head tilt using the left and right ear line."""
    left_ear = get_landmark_coordinates(landmarks, "left_ear")
    right_ear = get_landmark_coordinates(landmarks, "right_ear")

    if left_ear is None or right_ear is None:
        return None, "N/A"

    dx = right_ear[0] - left_ear[0]
    dy = right_ear[1] - left_ear[1]
    angle = math.degrees(math.atan2(dy, dx))
    direction = "left" if left_ear[1] < right_ear[1] else "right" if left_ear[1] > right_ear[1] else "neutral"
    return abs(angle), direction


def build_pose_analysis(landmarks):
    """Return a structured dictionary of detected landmarks and derived angles."""
    neck_angle = calculate_neck_angle(landmarks)
    shoulder_tilt_angle, shoulder_direction = calculate_shoulder_tilt(landmarks)
    spine_angle = calculate_spine_angle(landmarks)
    head_tilt_angle, head_direction = calculate_head_tilt(landmarks)

    return {
        "landmarks": landmarks,
        "angles": {
            "neck_angle": neck_angle,
            "shoulder_tilt": shoulder_tilt_angle,
            "spine_angle": spine_angle,
            "head_tilt": head_tilt_angle,
        },
        "directions": {
            "shoulder_tilt": shoulder_direction,
            "head_tilt": head_direction,
        },
    }


def draw_helper_visuals(frame, landmarks):
    """Draw circles and helper lines for the important landmarks."""
    if not landmarks:
        return

    important_points = [get_landmark_coordinates(landmarks, name) for name in IMPORTANT_LANDMARKS]
    if DRAW_LANDMARKS:
        for point in important_points:
            if point is None:
                continue
            cv2.circle(frame, point, IMPORTANT_LANDMARK_RADIUS, IMPORTANT_LANDMARK_COLOR, -1)

    if DRAW_HELPERS:
        left_shoulder = get_landmark_coordinates(landmarks, "left_shoulder")
        right_shoulder = get_landmark_coordinates(landmarks, "right_shoulder")
        left_hip = get_landmark_coordinates(landmarks, "left_hip")
        right_hip = get_landmark_coordinates(landmarks, "right_hip")
        nose = get_landmark_coordinates(landmarks, "nose")
        left_ear = get_landmark_coordinates(landmarks, "left_ear")
        right_ear = get_landmark_coordinates(landmarks, "right_ear")

        if left_shoulder and right_shoulder:
            cv2.line(frame, left_shoulder, right_shoulder, HELPER_LINE_COLOR, HELPER_LINE_THICKNESS)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_midpoint = (
                int((left_shoulder[0] + right_shoulder[0]) / 2),
                int((left_shoulder[1] + right_shoulder[1]) / 2),
            )
            hip_midpoint = (
                int((left_hip[0] + right_hip[0]) / 2),
                int((left_hip[1] + right_hip[1]) / 2),
            )
            cv2.line(frame, shoulder_midpoint, hip_midpoint, HELPER_LINE_COLOR, HELPER_LINE_THICKNESS)
        if nose and left_shoulder and right_shoulder:
            shoulder_midpoint = (
                int((left_shoulder[0] + right_shoulder[0]) / 2),
                int((left_shoulder[1] + right_shoulder[1]) / 2),
            )
            cv2.line(frame, nose, shoulder_midpoint, HELPER_LINE_COLOR, HELPER_LINE_THICKNESS)
        if left_ear and right_ear:
            cv2.line(frame, left_ear, right_ear, HELPER_LINE_COLOR, HELPER_LINE_THICKNESS)


def determine_posture(landmarks, angles):
    """Classify posture from the extracted landmarks and derived angles."""
    required_landmarks = {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    if not required_landmarks.issubset(landmarks.keys()):
        return "TRACKING LOST", None, False

    neck_angle = angles.get("neck_angle")
    shoulder_tilt = angles.get("shoulder_tilt")
    spine_angle = angles.get("spine_angle")
    head_tilt = angles.get("head_tilt")

    if neck_angle is not None and neck_angle > MAX_NECK_ANGLE:
        return "FORWARD HEAD", "head", True
    if spine_angle is not None and spine_angle > MAX_SPINE_INCLINATION:
        return "SLOUCHING", "spine", True
    if shoulder_tilt is not None and shoulder_tilt > MAX_SHOULDER_TILT:
        return "ROUNDED SHOULDERS", "shoulders", True

    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_midpoint = (
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
        )
        hip_midpoint = (
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
        )
        lean_offset = shoulder_midpoint[0] - hip_midpoint[0]
        if lean_offset < -MAX_LEAN_OFFSET:
            return "LEANING LEFT", "centerline", True
        if lean_offset > MAX_LEAN_OFFSET:
            return "LEANING RIGHT", "centerline", True

    if head_tilt is not None and head_tilt > MAX_HEAD_TILT:
        return "FORWARD HEAD", "head", True

    return "GOOD POSTURE", None, True


def apply_status_smoothing(pose_detector, new_status):
    """Avoid flickering by smoothing status changes across several frames."""
    if not hasattr(pose_detector, "_posture_state"):
        pose_detector._posture_state = {"current_status": "GOOD POSTURE", "streak": 0}

    state = pose_detector._posture_state
    if new_status == "TRACKING LOST":
        state["current_status"] = new_status
        state["streak"] = 0
        return new_status

    if new_status == state["current_status"]:
        state["streak"] = 0
        return new_status

    state["streak"] += 1
    if state["streak"] >= STATUS_HYSTERESIS_FRAMES:
        state["current_status"] = new_status
        state["streak"] = 0
    return state["current_status"]


def get_status_color(status):
    """Return a display color for a posture status."""
    if status == "GOOD POSTURE":
        return STATUS_GOOD_COLOR
    if status in {"FORWARD HEAD", "SLOUCHING", "ROUNDED SHOULDERS"}:
        return STATUS_ALERT_COLOR
    if status in {"LEANING LEFT", "LEANING RIGHT"}:
        return STATUS_WARNING_COLOR
    if status == "TRACKING LOST":
        return STATUS_NEUTRAL_COLOR
    return STATUS_NEUTRAL_COLOR


def initialize_dashboard_state():
    """Create the live dashboard metrics state used by the overlay."""
    return {
        "session_start_time": time.time(),
        "last_metrics_update": 0.0,
        "fps": 0.0,
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "frame_count": 0,
        "current_status": "GOOD POSTURE",
        "posture_durations": {"good": 0.0, "bad": 0.0},
    }


def initialize_session_analytics(alert_duration_seconds=ALERT_DURATION_SECONDS):
    """Create the session analytics state used for alerts and reporting."""
    return {
        "session_start_time": time.time(),
        "alerts_triggered": 0,
        "posture_corrections": 0,
        "posture_durations": {"good": 0.0, "bad": 0.0},
        "current_status": "GOOD POSTURE",
        "active_alert": False,
        "bad_posture_timer": 0.0,
        "longest_good_streak": 0.0,
        "longest_bad_streak": 0.0,
        "current_good_streak": 0.0,
        "current_bad_streak": 0.0,
        "alert_duration_seconds": alert_duration_seconds,
        "last_alert_time": None,
    }


def initialize_calibration_state():
    """Create state for a user-specific posture baseline."""
    return {
        "is_calibrated": False,
        "baseline": {
            "neck_angle": 0.0,
            "spine_angle": 0.0,
            "shoulder_tilt": 0.0,
            "head_tilt": 0.0,
        },
        "samples": 0,
        "calibration_start_time": None,
        "calibration_active": False,
        "calibration_message": "Press C to calibrate",
    }


def update_calibration_state(calibration_state, angles):
    """Collect posture samples until a baseline is established."""
    if not calibration_state.get("calibration_active") and not calibration_state.get("is_calibrated"):
        calibration_state["calibration_active"] = True

    if not calibration_state.get("calibration_active"):
        return calibration_state

    calibration_state["samples"] += 1
    for key in calibration_state["baseline"]:
        calibration_state["baseline"][key] += angles.get(key, 0.0)

    if calibration_state["samples"] >= CALIBRATION_SAMPLE_COUNT:
        for key in calibration_state["baseline"]:
            calibration_state["baseline"][key] /= CALIBRATION_SAMPLE_COUNT
        calibration_state["is_calibrated"] = True
        calibration_state["calibration_active"] = False
        calibration_state["calibration_message"] = "Calibration complete"

    return calibration_state


def compute_posture_score(deviation):
    """Convert a posture deviation score into a 0-100 quality score."""
    deviation = max(0.0, min(float(deviation), 100.0))
    if deviation <= 10.0:
        return 100
    score = int(round(100.0 - deviation))
    return max(0, min(score, 100))


def smooth_score(current_score, new_score):
    """Smooth a posture score to reduce rapid fluctuations."""
    if current_score is None:
        return new_score
    return int(round(current_score * (1.0 - SCORE_SMOOTHING_FACTOR) + new_score * SCORE_SMOOTHING_FACTOR))


def get_posture_score_label(score):
    """Return a label for a posture score."""
    if score >= 95:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Very Poor"


def get_recommendation(status):
    """Return a single posture recommendation for the current posture issue."""
    recommendations = {
        "FORWARD HEAD": "Bring your head slightly backward.",
        "SLOUCHING": "Sit upright and engage your core.",
        "ROUNDED SHOULDERS": "Roll your shoulders back.",
        "LEANING LEFT": "Center your torso.",
        "LEANING RIGHT": "Center your torso.",
    }
    return recommendations.get(status)


def update_dashboard_state(state, status, frame_delta):
    """Update dashboard metrics and posture duration counters."""
    state["frame_count"] += 1
    state["current_status"] = status

    if status == "GOOD POSTURE":
        state["posture_durations"]["good"] += frame_delta
    elif status != "TRACKING LOST":
        state["posture_durations"]["bad"] += frame_delta

    return state


def update_session_analytics(analytics_state, status, frame_delta):
    """Update posture durations, streaks, corrections, and alert state."""
    analytics_state["current_status"] = status

    if status == "GOOD POSTURE":
        analytics_state["posture_durations"]["good"] += frame_delta
        analytics_state["current_good_streak"] += frame_delta
        analytics_state["current_bad_streak"] = 0.0
        analytics_state["longest_good_streak"] = max(analytics_state["longest_good_streak"], analytics_state["current_good_streak"])
        analytics_state["bad_posture_timer"] = 0.0

        if analytics_state["active_alert"]:
            analytics_state["posture_corrections"] += 1
            analytics_state["active_alert"] = False
    elif status != "TRACKING LOST":
        analytics_state["posture_durations"]["bad"] += frame_delta
        analytics_state["current_bad_streak"] += frame_delta
        analytics_state["current_good_streak"] = 0.0
        analytics_state["longest_bad_streak"] = max(analytics_state["longest_bad_streak"], analytics_state["current_bad_streak"])

        if ENABLE_ALERTS and not analytics_state["active_alert"]:
            analytics_state["bad_posture_timer"] += frame_delta
            if analytics_state["bad_posture_timer"] >= analytics_state["alert_duration_seconds"]:
                analytics_state["active_alert"] = True
                analytics_state["alerts_triggered"] += 1
                analytics_state["last_alert_time"] = time.time()
    else:
        analytics_state["current_good_streak"] = 0.0
        analytics_state["current_bad_streak"] = 0.0
        analytics_state["bad_posture_timer"] = 0.0

    return analytics_state


def save_screenshot(frame, output_dir=SCREENSHOT_DIRECTORY):
    """Save the current frame as a timestamped image file."""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(output_dir) / f"posture_{timestamp}.png"
        success = cv2.imwrite(str(path), frame)
        if not success:
            raise OSError(f"Failed to write screenshot: {path}")
        return str(path)
    except Exception as exc:
        print(f"[Warning] Screenshot capture failed: {exc}")
        return None


def export_session_csv(analytics_state, output_dir=REPORT_DIRECTORY, filename_format=CSV_FILENAME_FORMAT):
    """Export the completed session analytics to a CSV report."""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime(filename_format)
        report_path = Path(output_dir) / f"session_{timestamp}.csv"
        total_duration = analytics_state.get("posture_durations", {}).get("good", 0.0) + analytics_state.get("posture_durations", {}).get("bad", 0.0)
        good_percentage = 0.0 if total_duration <= 0 else (analytics_state["posture_durations"]["good"] / total_duration) * 100.0
        bad_percentage = 100.0 - good_percentage

        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerow(["date", datetime.now().strftime("%Y-%m-%d")])
            writer.writerow(["start_time", datetime.fromtimestamp(analytics_state["session_start_time"]).strftime("%H:%M:%S")])
            writer.writerow(["end_time", datetime.now().strftime("%H:%M:%S")])
            writer.writerow(["total_session_duration", f"{total_duration:.2f}s"])
            writer.writerow(["good_posture_percentage", f"{good_percentage:.2f}%"])
            writer.writerow(["bad_posture_percentage", f"{bad_percentage:.2f}%"])
            writer.writerow(["posture_corrections", analytics_state["posture_corrections"]])
            writer.writerow(["alerts_triggered", analytics_state["alerts_triggered"]])
            writer.writerow(["longest_good_streak", f"{analytics_state['longest_good_streak']:.2f}s"])
            writer.writerow(["longest_bad_streak", f"{analytics_state['longest_bad_streak']:.2f}s"])
            writer.writerow(["average_posture_status", analytics_state["current_status"]])

        return str(report_path)
    except Exception as exc:
        print(f"[Warning] CSV export failed: {exc}")
        return None


def get_system_metrics():
    """Return a lightweight snapshot of CPU and memory usage."""
    try:
        cpu_percent = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    except OSError:
        cpu_percent = 0.0

    try:
        import psutil

        ram = psutil.virtual_memory()
        return float(cpu_percent), float(ram.percent)
    except Exception:
        return float(cpu_percent), 0.0


def draw_status_panel(frame, status):
    """Render a small posture status panel on the frame."""
    color = get_status_color(status)
    cv2.rectangle(frame, (10, 10), (330, 60), STATUS_PANEL_BG_COLOR, -1)
    cv2.rectangle(frame, (10, 10), (330, 60), color, 2)
    cv2.putText(frame, "CURRENT STATUS", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_debug_overlay(frame, analysis):
    """Render live angle information on the frame."""
    if not DRAW_DEBUG_INFO:
        return

    angles = analysis.get("angles", {})
    directions = analysis.get("directions", {})

    def format_value(value):
        return "N/A" if value is None else f"{value:.1f}"

    lines = [
        f"Neck Angle: {format_value(angles.get('neck_angle'))}",
        f"Shoulder Tilt: {format_value(angles.get('shoulder_tilt'))} ({directions.get('shoulder_tilt', 'N/A')})",
        f"Spine Angle: {format_value(angles.get('spine_angle'))}",
        f"Head Tilt: {format_value(angles.get('head_tilt'))} ({directions.get('head_tilt', 'N/A')})",
    ]

    overlay_height = 120
    overlay_width = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 75), (10 + overlay_width, 75 + overlay_height), DEBUG_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 100
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEBUG_TEXT_COLOR, 1, cv2.LINE_AA)
        y += 22


def draw_dashboard_overlay(frame, analysis, dashboard_state, frame_shape, calibration_state=None, recommendation=None):
    """Render a compact dashboard overlay with metrics and session stats."""
    width, height = frame_shape[1], frame_shape[0]
    panel_x, panel_y = width - 320, 10
    panel_w, panel_h = 300, 245
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), DASHBOARD_BG_COLOR, -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), DASHBOARD_BORDER_COLOR, 2)

    title = "POSTURE DASHBOARD"
    cv2.putText(frame, title, (panel_x + 16, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DASHBOARD_TEXT_COLOR, 1, cv2.LINE_AA)

    status = analysis.get("posture", {}).get("status", "GOOD POSTURE")
    status_color = get_status_color(status)
    cv2.putText(frame, f"Status: {status}", (panel_x + 16, panel_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    elapsed = time.time() - dashboard_state["session_start_time"]
    minutes, seconds = divmod(int(elapsed), 60)
    cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (panel_x + 16, panel_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_INFO_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {dashboard_state['fps']:.1f}", (panel_x + 16, panel_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"CPU: {dashboard_state['cpu_percent']:.1f}%", (panel_x + 16, panel_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"RAM: {dashboard_state['ram_percent']:.1f}%", (panel_x + 16, panel_y + 156), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Res: {frame_shape[1]}x{frame_shape[0]}", (panel_x + 16, panel_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_INFO_COLOR, 1, cv2.LINE_AA)

    good_time = dashboard_state["posture_durations"]["good"]
    bad_time = dashboard_state["posture_durations"]["bad"]
    score = analysis.get("posture", {}).get("score", 0)
    cv2.putText(frame, f"Score: {score}/100", (panel_x + 16, panel_y + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHBOARD_INFO_COLOR, 1, cv2.LINE_AA)
    calibration_text = "Calibrated" if calibration_state and calibration_state.get("is_calibrated") else "Calibrating"
    cv2.putText(frame, f"Calibration: {calibration_text}", (panel_x + 16, panel_y + 228), cv2.FONT_HERSHEY_SIMPLEX, 0.47, DASHBOARD_MUTED_COLOR, 1, cv2.LINE_AA)
    if recommendation:
        cv2.putText(frame, f"Rec: {recommendation}", (panel_x + 16, panel_y + 252), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DASHBOARD_TEXT_COLOR, 1, cv2.LINE_AA)


def highlight_body_region(frame, landmarks, highlight_region, status):
    """Visually emphasize the region linked to the current posture issue."""
    if not highlight_region:
        return

    color = get_status_color(status)
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    nose = landmarks.get("nose")
    left_ear = landmarks.get("left_ear")
    right_ear = landmarks.get("right_ear")

    if highlight_region == "head" and nose and left_ear and right_ear:
        cv2.circle(frame, nose, 8, color, 2)
        cv2.line(frame, left_ear, right_ear, color, 2)
    elif highlight_region == "spine" and left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_midpoint = (
            int((left_shoulder[0] + right_shoulder[0]) / 2),
            int((left_shoulder[1] + right_shoulder[1]) / 2),
        )
        hip_midpoint = (
            int((left_hip[0] + right_hip[0]) / 2),
            int((left_hip[1] + right_hip[1]) / 2),
        )
        cv2.line(frame, shoulder_midpoint, hip_midpoint, color, 3)
    elif highlight_region == "shoulders" and left_shoulder and right_shoulder:
        cv2.circle(frame, left_shoulder, 8, color, 2)
        cv2.circle(frame, right_shoulder, 8, color, 2)
        cv2.line(frame, left_shoulder, right_shoulder, color, 2)
    elif highlight_region == "centerline" and left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_midpoint = (
            int((left_shoulder[0] + right_shoulder[0]) / 2),
            int((left_shoulder[1] + right_shoulder[1]) / 2),
        )
        hip_midpoint = (
            int((left_hip[0] + right_hip[0]) / 2),
            int((left_hip[1] + right_hip[1]) / 2),
        )
        cv2.line(frame, shoulder_midpoint, hip_midpoint, color, 3)


def detect_pose(frame, pose_detector, dashboard_state=None, analytics_state=None, frame_delta=None, calibration_state=None, posture_score=None):
    """Run pose detection, draw the skeleton, and return landmark/angle analysis."""
    if dashboard_state is None:
        dashboard_state = initialize_dashboard_state()
    if analytics_state is None:
        analytics_state = initialize_session_analytics()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    if not hasattr(pose_detector, "_timestamp_ms"):
        pose_detector._timestamp_ms = 0
    pose_detector._timestamp_ms += 33

    try:
        results = pose_detector.detect_for_video(image, pose_detector._timestamp_ms)
    except Exception as exc:
        print(f"[Warning] Pose detection failed: {exc}")
        analysis = build_pose_analysis({})
        analysis["posture"] = {
            "status": "TRACKING LOST",
            "highlight_region": None,
            "tracking_ok": False,
        }
        draw_status_panel(frame, "TRACKING LOST")
        draw_debug_overlay(frame, analysis)
        return frame, analysis, 0

    landmarks = {}
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks[0]
        raw_landmarks = extract_landmarks(pose_landmarks, frame.shape)
        if not hasattr(pose_detector, "_smoothed_landmarks"):
            pose_detector._smoothed_landmarks = {}
        pose_detector._smoothed_landmarks = smooth_landmarks(pose_detector._smoothed_landmarks, raw_landmarks)
        landmarks = pose_detector._smoothed_landmarks

        if DRAW_LANDMARKS:
            for landmark in pose_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

        if DRAW_CONNECTIONS:
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                    continue
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]
                x1 = int(start.x * frame.shape[1])
                y1 = int(start.y * frame.shape[0])
                x2 = int(end.x * frame.shape[1])
                y2 = int(end.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), CONNECTION_COLOR, CONNECTION_THICKNESS)

    analysis = build_pose_analysis(landmarks)
    posture_status, highlight_region, tracking_ok = determine_posture(landmarks, analysis["angles"])
    angles = analysis.get("angles", {})
    if calibration_state is not None and calibration_state.get("calibration_active"):
        update_calibration_state(calibration_state, {
            "neck_angle": angles.get("neck_angle") or 0.0,
            "spine_angle": angles.get("spine_angle") or 0.0,
            "shoulder_tilt": angles.get("shoulder_tilt") or 0.0,
            "head_tilt": angles.get("head_tilt") or 0.0,
        })

    baseline = calibration_state.get("baseline", {}) if calibration_state else {}
    if calibration_state and calibration_state.get("is_calibrated"):
        deviation = 0.0
        for key, value in baseline.items():
            if key in angles and angles[key] is not None:
                deviation += abs((angles[key] or 0.0) - value)
        deviation = min(deviation * 2.0, 100.0)
        computed_score = compute_posture_score(deviation)
    else:
        computed_score = 100 if posture_status == "GOOD POSTURE" else 60

    if posture_score is None:
        posture_score = computed_score
    else:
        posture_score = smooth_score(posture_score, computed_score)
    smoothed_status = apply_status_smoothing(pose_detector, posture_status)
    analysis["posture"] = {
        "status": smoothed_status,
        "highlight_region": highlight_region if tracking_ok else None,
        "tracking_ok": tracking_ok,
        "score": posture_score,
        "recommendation": get_recommendation(smoothed_status),
    }

    draw_helper_visuals(frame, landmarks)
    highlight_body_region(frame, landmarks, analysis["posture"]["highlight_region"], smoothed_status)
    draw_status_panel(frame, smoothed_status)
    draw_debug_overlay(frame, analysis)

    if frame_delta is None:
        frame_delta = 1.0 / max(dashboard_state.get("fps", 24.0), 1e-6)
    if dashboard_state is not None:
        now = time.time()
        delta_seconds = max(now - dashboard_state.get("last_metrics_update", now), 0.0)
        if delta_seconds >= METRICS_UPDATE_SECONDS:
            dashboard_state["last_metrics_update"] = now
            dashboard_state["fps"] = 1.0 / max(delta_seconds, 1e-6)
            dashboard_state["cpu_percent"], dashboard_state["ram_percent"] = get_system_metrics()
        update_dashboard_state(dashboard_state, smoothed_status, frame_delta)
        draw_dashboard_overlay(frame, analysis, dashboard_state, frame.shape, calibration_state, analysis["posture"].get("recommendation"))

    if analytics_state is not None:
        update_session_analytics(analytics_state, smoothed_status, frame_delta)

    return frame, analysis, posture_score
