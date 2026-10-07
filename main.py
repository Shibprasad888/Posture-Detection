"""Real-time posture detection application entry point."""

import time

import cv2

from config import CAMERA_INDEX, ENABLE_SCREENSHOTS, FRAME_HEIGHT, FRAME_WIDTH, TARGET_FPS
from posture import (
    detect_pose,
    export_session_csv,
    initialize_calibration_state,
    initialize_dashboard_state,
    initialize_pose_detector,
    initialize_session_analytics,
    save_screenshot,
)
from ui import draw_ui


def main() -> None:
    """Open the webcam, run pose detection, and display the result."""
    capture = cv2.VideoCapture(CAMERA_INDEX)

    if not capture.isOpened():
        raise RuntimeError("Unable to open the default webcam.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    try:
        pose_detector = initialize_pose_detector()
    except Exception as exc:
        capture.release()
        raise RuntimeError(f"Unable to initialize pose detector: {exc}") from exc

    dashboard_state = initialize_dashboard_state()
    analytics_state = initialize_session_analytics()
    calibration_state = initialize_calibration_state()
    session_ended = False
    last_frame_time = time.time()
    posture_score = None
    break_timer = 0.0
    score_history = []

    while True:
        success, frame = capture.read()
        if not success:
            print("Camera stream ended or disconnected.")
            break

        now = time.time()
        frame_delta = max(now - last_frame_time, 1e-3)
        last_frame_time = now

        processed_frame, analysis, posture_score = detect_pose(frame, pose_detector, dashboard_state, analytics_state, frame_delta, calibration_state, posture_score)
        if posture_score is not None:
            score_history.append(posture_score)

        if analytics_state.get("active_alert"):
            cv2.putText(processed_frame, "ALERT: Maintain better posture", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        final_frame = draw_ui(processed_frame)

        cv2.imshow("Posture Detection", final_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            session_ended = True
            break
        if key == ord("c"):
            calibration_state["calibration_active"] = True
            calibration_state["samples"] = 0
            calibration_state["baseline"] = {"neck_angle": 0.0, "spine_angle": 0.0, "shoulder_tilt": 0.0, "head_tilt": 0.0}
            calibration_state["calibration_start_time"] = time.time()
            calibration_state["calibration_message"] = "Hold ideal posture for 5 seconds"

        if calibration_state.get("calibration_active") and calibration_state.get("calibration_start_time"):
            elapsed = time.time() - calibration_state["calibration_start_time"]
            if elapsed >= 5.0:
                calibration_state["calibration_active"] = False
                calibration_state["is_calibrated"] = calibration_state["samples"] >= 1
                calibration_state["calibration_message"] = "Calibration complete"
        if key == ord("s"):
            if ENABLE_SCREENSHOTS:
                save_screenshot(final_frame)
            else:
                print("Screenshot capture is disabled in config.")

    capture.release()
    cv2.destroyAllWindows()

    if not session_ended:
        print("Session ended unexpectedly.")

    total_duration = analytics_state["posture_durations"]["good"] + analytics_state["posture_durations"]["bad"]
    good_percentage = 0.0 if total_duration <= 0 else (analytics_state["posture_durations"]["good"] / total_duration) * 100.0
    bad_percentage = 100.0 - good_percentage

    print("\nSession Summary")
    print("-" * 24)
    print(f"Total Duration: {total_duration:.1f}s")
    print(f"Good Posture: {analytics_state['posture_durations']['good']:.1f}s")
    print(f"Bad Posture: {analytics_state['posture_durations']['bad']:.1f}s")
    print(f"Good %: {good_percentage:.1f}%")
    print(f"Bad %: {bad_percentage:.1f}%")
    print(f"Corrections: {analytics_state['posture_corrections']}")
    print(f"Alerts Triggered: {analytics_state['alerts_triggered']}")
    print(f"Longest Good Streak: {analytics_state['longest_good_streak']:.1f}s")
    print(f"Longest Bad Streak: {analytics_state['longest_bad_streak']:.1f}s")
    if score_history:
        print(f"Average Score: {sum(score_history) / len(score_history):.1f}")
        print(f"Best Score: {max(score_history)}")
        print(f"Lowest Score: {min(score_history)}")

    export_session_csv(analytics_state)


if __name__ == "__main__":
    main()
