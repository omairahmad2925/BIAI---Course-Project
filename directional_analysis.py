'''
Omair Ahmad - 251443603
Yukti. - 251400558
------------------------
AI-Powered Real-Time Suspicious Behavior Detection in CCTV Footage
------------------------
This code implements Directional Analysis which is a part of the System Architecture's, Erratic Trajectory Module.
'''

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class HorizontalDirectionTracker:
    def __init__(self, fps, confirm_frames=8, suspicion_increase=0.4, decay_rate=0.05):
        self.fps = fps
        self.confirm_frames = confirm_frames
        self.suspicion_increase = suspicion_increase
        self.decay_per_frame = decay_rate / fps
        self.positions = deque(maxlen=confirm_frames)

        self.last_direction = None
        self.suspicion = 0.0
        self.reason = "Initializing"

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0

        self.positions.append(center_x)

        if len(self.positions) < self.confirm_frames:
            return self.suspicion, "Stabilizing..."

        avg_diff = np.mean(np.diff(self.positions))

        if abs(avg_diff) < 0.5:
            direction = 0
        else:
            direction = np.sign(avg_diff)

        suspicion_changed = False
        if direction and self.last_direction and direction != self.last_direction:
            self.suspicion = min(1.0, self.suspicion + self.suspicion_increase)
            self.reason = "Direction Reversal"
            suspicion_changed = True
        else:
            self.suspicion = max(0.0, self.suspicion - self.decay_per_frame)
            if direction == -1:
                self.reason = "Moving Left"
            elif direction == 1:
                self.reason = "Moving Right"
            else:
                self.reason = "Stationary"

        if direction != 0:
            self.last_direction = direction

        return self.suspicion, self.reason

def main():
    video_path = "/Users/omairahmad_/Desktop/BAI - Project/Videos/check.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("horizontal_direction_suspicion.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    model = YOLO("yolo11x.pt")
    trackers = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, conf=0.4, device='mps', classes=[0])
        annotated_frame = frame.copy()

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, pid in zip(boxes, ids):
                tracker = trackers.setdefault(pid, HorizontalDirectionTracker(fps))

                suspicion, reason = tracker.update(bbox)

                color = (0, 0, 255) if suspicion >= 0.8 else \
                        (0, 165, 255) if suspicion >= 0.4 else \
                        (0, 255, 0)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID {pid}: Susp {suspicion:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(annotated_frame, reason,
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        out.write(annotated_frame)
        cv2.imshow("Horizontal Direction Suspicion", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Saved as horizontal_direction_suspicion.mp4")

if __name__ == "__main__":
    main()