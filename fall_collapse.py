'''
Omair Ahmad - 251443603
Yukti. - 251400558
------------------------
AI-Powered Real-Time Suspicious Behavior Detection in CCTV Footage
------------------------
This code implements fall or collapse detection which is a part of the System Architecture's, Fall Detection Module.
'''

import os
import numpy as np
import cv2
from ultralytics import YOLO

ANGLE_THRESHOLD        = 35
FALL_ASPECT_RATIO      = 1.0
HEIGHT_DROP_THRESHOLD  = 0.8
VELOCITY_THRESHOLD     = 15
HISTORY_LENGTH         = 5
FALL_FRAMES_REQUIRED   = 3
COLLAPSE_TIME          = 7

model      = YOLO("yolo11m-pose.pt")
video_path = "/Users/omairahmad_/Desktop/BAI - Project/Codes/fall.mp4"
cap        = cv2.VideoCapture(video_path)

fps    = cap.get(cv2.CAP_PROP_FPS)
w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter("processed_video.mp4", fourcc, fps, (w, h))

keypoint_labels = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle"
]

height_histories    = {}
prev_torso_centers  = {}
fall_counters       = {}
fall_start_frames   = {}
alerted             = set()

def send_emergency_alert(pid):
    print(f"!!! EMERGENCY: Person {pid} has been down for >{COLLAPSE_TIME}s !!!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        results = model.track(frame, device="mps", persist=True)
        result  = results[0]
        ann     = result.orig_img if hasattr(result, 'orig_img') else frame.copy()
        boxes, kpts = result.boxes, result.keypoints

        if boxes.id is None:
            cv2.imshow("Fall/Collapse Detection", ann)
            out.write(ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for box, pid_f, pts in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy(),
            kpts.data.cpu().numpy()
        ):
            pid = int(pid_f)
            x1,y1,x2,y2 = map(int, box)
            bw, bh = x2-x1, y2-y1
            aspect = bw / float(bh)

            hist = height_histories.setdefault(pid, [])
            hist.append(bh)
            if len(hist) > HISTORY_LENGTH:
                hist.pop(0)
            if len(hist) > 1:
                median_prev = np.median(hist[:-1])
                height_drop = (bh / float(median_prev)) < HEIGHT_DROP_THRESHOLD
            else:
                height_drop = False

            sh_vis  = pts[[5,6],2]; hip_vis = pts[[11,12],2]
            fall_by_pose = fall_by_velocity = False
            if np.all(sh_vis>0) and np.all(hip_vis>0):
                ls, rs = pts[5][:2], pts[6][:2]
                lh, rh = pts[11][:2], pts[12][:2]
                shoulder_mid = (ls + rs) / 2
                hip_mid      = (lh + rh) / 2
                dx, dy = shoulder_mid - hip_mid
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                fall_by_pose = (angle < ANGLE_THRESHOLD)

                torso_mid = (shoulder_mid + hip_mid) / 2
                prev_mid  = prev_torso_centers.get(pid, torso_mid)
                vel_down  = prev_mid[1] - torso_mid[1]
                prev_torso_centers[pid] = torso_mid
                fall_by_velocity = (vel_down > VELOCITY_THRESHOLD)

            fall_by_bbox = (aspect > FALL_ASPECT_RATIO)

            is_fall = fall_by_pose or fall_by_bbox or height_drop or fall_by_velocity

            count = fall_counters.get(pid, 0)
            count = count + 1 if is_fall else 0
            fall_counters[pid] = count

            if count == FALL_FRAMES_REQUIRED:
                fall_start_frames[pid] = current_frame_idx

            label = f"ID {pid}"
            color = (0,0,255)

            if count >= FALL_FRAMES_REQUIRED:
                elapsed_frames = current_frame_idx - fall_start_frames.get(pid, current_frame_idx)
                elapsed_sec    = elapsed_frames / fps
                sec            = int(elapsed_sec)

                if elapsed_sec < COLLAPSE_TIME:
                    label = f"ID {pid} - FALL , ({sec}s < {COLLAPSE_TIME}s)"
                    color = (0,165,255)
                else:
                    label = f"ID {pid} - COLLAPSE ALERT , ({sec}s >= {COLLAPSE_TIME}s)"
                    color = (0,255,255)
                    if pid not in alerted:
                        send_emergency_alert(pid)
                        alerted.add(pid)

            cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
            cv2.putText(ann, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for idx, kp in enumerate(pts):
                if idx < 5:
                    continue
                x,y,v = kp
                if v:
                    cv2.circle(ann, (int(x),int(y)), 3, color, -1)
                    cv2.putText(ann, keypoint_labels[idx],
                                (int(x)+5, int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Fall/Collapse Detection", ann)
        out.write(ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processed video saved at:", os.path.abspath("processed_video.mp4"))