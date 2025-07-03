import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def process_squat_video(input_path, output_path):
    print(f"[INFO] Starting squat video processing for: {input_path}")
    raw_path = output_path.replace('.mp4', '_raw.mp4')

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    squat_counter = 0
    squat_stage = None
    rep_start_time = None
    rep_durations = []
    min_knee_angles = []
    knees_caving_in_reps = 0
    reps_below_parallel = 0
    form_issues = []
    rep_min_knee_angle = None
    rep_knees_caving_in = False
    all_rep_issues = []
    latest_rep_feedback = ""
    last_rep_time = None
    show_duration = 0.9  # seconds
    blank_duration = 0.7  # seconds
    frame_count = 0
    sample_rate = 3  # Process every 3rd frame

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("[INFO] MediaPipe Pose model loaded.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frame sampling - process every 3rd frame
            if frame_count % sample_rate != 0:
                frame_count += 1
                # Still write the frame to output video for smooth playback
                out.write(frame)
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Angles
                knee_angle = calculate_angle(hip, knee, ankle)

                # Track min knee angle for this rep
                if rep_min_knee_angle is None or knee_angle < rep_min_knee_angle:
                    rep_min_knee_angle = knee_angle

                # Detect knees caving in (knee x < hip x and knee x < ankle x by a margin)
                if (knee[0] < hip[0] - 0.03) and (knee[0] < ankle[0] - 0.03):
                    rep_knees_caving_in = True

                # Rep detection logic
                if knee_angle > 160:
                    if squat_stage == "down":
                        squat_stage = "up"
                        # Rep finished
                        squat_counter += 1
                        print(f"[INFO] Rep {squat_counter} finished. Min knee angle: {rep_min_knee_angle:.2f}")
                        min_knee_angles.append(rep_min_knee_angle)
                        rep_issues = []
                        feedback_reasons = []
                        if rep_min_knee_angle < 100:
                            reps_below_parallel += 1
                        else:
                            rep_issues.append("shallow_depth")
                            feedback_reasons.append("go deeper")
                        if rep_knees_caving_in:
                            knees_caving_in_reps += 1
                            rep_issues.append("knees_in")
                            feedback_reasons.append("knees in")
                        all_rep_issues.append(rep_issues)
                        # Tempo
                        if rep_start_time is not None:
                            rep_end_time = time.time()
                            rep_durations.append(rep_end_time - rep_start_time)
                            rep_start_time = None
                        # Set feedback text for this rep
                        if not feedback_reasons:
                            latest_rep_feedback = "Good rep"
                        else:
                            latest_rep_feedback = "Bad rep - " + ", ".join(feedback_reasons)
                        last_rep_time = time.time()  # Mark the time of this rep
                        # Reset per-rep trackers
                        rep_min_knee_angle = None
                        rep_knees_caving_in = False
                    else:
                        squat_stage = "up"
                if knee_angle < 90 and squat_stage == 'up':
                    squat_stage = "down"
                    # Rep started
                    rep_start_time = time.time()

            except Exception as e:
                pass

            # --- BLINKING ANNOTATION LOGIC ---
            show_text = False
            if last_rep_time is not None:
                elapsed = time.time() - last_rep_time
                if elapsed < show_duration:
                    show_text = True  # Show feedback
                elif elapsed < show_duration + blank_duration:
                    show_text = False  # Hide feedback (blank)
                else:
                    show_text = False  # Keep hidden until next rep
            if show_text and latest_rep_feedback:
                cv2.putText(
                    image,
                    latest_rep_feedback,
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if latest_rep_feedback.startswith("Bad rep") else (0, 200, 0),
                    3,
                    cv2.LINE_AA
                )

            out.write(image)
            frame_count += 1

    cap.release()
    out.release()

    # 🔁 Step 2: Convert to H.264 using ffmpeg
    print("🎞 Converting to H.264 using ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', raw_path,
        '-vcodec', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-acodec', 'aac',
        output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    os.remove(raw_path)
    print("✅ H.264 conversion complete")
    print(f"[INFO] Processing complete. Total squats counted: {squat_counter}")

    # Aggregate stats
    squat_count = squat_counter
    reps_below_parallel = reps_below_parallel
    bad_reps = knees_caving_in_reps
    form_issues = list(set([issue for rep in all_rep_issues for issue in rep]))
    rep_time = {
        "average": round(float(np.mean(rep_durations)), 1) if rep_durations else 0.0,
        "fastest": round(float(np.min(rep_durations)), 1) if rep_durations else 0.0,
        "slowest": round(float(np.max(rep_durations)), 1) if rep_durations else 0.0
    }

    return {
        "squat_count": squat_count,
        "reps_below_parallel": reps_below_parallel,
        "bad_reps": bad_reps,
        "form_issues": form_issues,
        "rep_time": rep_time
    }
