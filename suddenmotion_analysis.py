'''
Omair Ahmad - 251443603
Yukti. - 251400558
------------------------
AI-Powered Real-Time Suspicious Behavior Detection in CCTV Footage
------------------------
This code implements Sudden motion analysis which is a part of the System Architecture's, Erratic Trajectory Module.
'''

import os, cv2, numpy as np
from collections import deque
from ultralytics import YOLO

class DispChangeDetector:
    def __init__(
        self,
        fps: float,
        disp_window: int = 10,
        ratio_thresh: float = 2.5,
        min_disp: float = 1.0,
        up_rate: float = 1.0,
        down_rate: float = 0.05,
        stand_thresh: float = 1.0,
    ):
        self.dt = 1.0 / fps
        self.disp_hist = deque(maxlen=disp_window)
        self.last_cent = None

        self.ratio_th = ratio_thresh
        self.min_disp = min_disp
        self.up_rate = up_rate
        self.down_rate = down_rate
        self.stand_th = stand_thresh
        self.window = disp_window

        self.suspicion = 0.0
        self.high_lock = False

        self.disp_now = 0.0
        self.base_disp = 0.0
        self.ratio = 0.0
        self.weight = 0.0
        self.reason = "init"

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if self.last_cent is None:
            self.last_cent = (cx, cy)
            return self.suspicion

        dx, dy = cx - self.last_cent[0], cy - self.last_cent[1]
        disp_now = np.hypot(dx, dy)
        base_disp = np.median(self.disp_hist) if self.disp_hist else 0.0

        self.disp_hist.append(disp_now)
        self.last_cent = (cx, cy)

        if (
            len(self.disp_hist) == self.window
            and base_disp < self.stand_th
            and disp_now >= self.stand_th
        ):
            self.suspicion = 1.0
            self.high_lock = True
            self._diag(disp_now, base_disp, 99.0, 1.0, "stand→run")
            return self.suspicion

        if self.high_lock:
            if disp_now < self.stand_th:
                self.high_lock = False
            self._diag(disp_now, base_disp, 0.0, 0.0, "latched")
            return self.suspicion

        baseline = max(base_disp, self.min_disp)
        ratio = disp_now / baseline
        weight = np.clip((ratio - 1.0) / (self.ratio_th - 1.0), 0.0, 1.0)

        if disp_now > self.min_disp and ratio >= self.ratio_th:
            self.suspicion += self.up_rate * weight * self.dt
            reason = "jump"
        else:
            self.suspicion -= self.down_rate * self.dt
            reason = "steady"

        self.suspicion = float(np.clip(self.suspicion, 0.0, 1.0))
        self._diag(disp_now, base_disp, ratio, weight, reason)
        return self.suspicion

    def _diag(self, disp_now, base_disp, ratio, weight, reason):
        self.disp_now = disp_now
        self.base_disp = base_disp
        self.ratio = ratio
        self.weight = weight
        self.reason = reason

def main():
    video_path = "/Users/omairahmad_/Desktop/BAI - Project/Videos/run.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("disp_change_suspicion.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO("yolo11s.pt")
    detectors = {}

    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    ORANGE = (0, 165, 255)
    RED = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.track(frame, device="cpu", persist=True, conf=0.7, classes=[0])
        vis = res[0].orig_img if hasattr(res[0], "orig_img") else frame.copy()

        if res and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy()
            classes = res[0].boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), pid_f, cls in zip(boxes, ids, classes):
                if cls != 0:
                    continue

                pid = int(pid_f)
                if pid not in detectors:
                    detectors[pid] = DispChangeDetector(fps)

                det = detectors[pid]
                score = det.update((x1, y1, x2, y2))

                box_col = YELLOW
                label_text = ""
                label_color = YELLOW
                if score >= 0.85:
                    box_col = RED
                    label_text = "HIGH SUSPICION"
                    label_color = RED
                elif score >= 0.60:
                    label_text = "LOW SUSPICION"
                    label_color = ORANGE
                elif score < 0.30:
                    label_text = "MIN SUSPICION"
                    label_color = BLUE

                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), box_col, 2)

                if label_text:
                    cv2.putText(vis, label_text, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

                cv2.putText(vis, f"{score:.2f}", (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

                y0 = int(y2) + 45
                cv2.putText(vis, f"Disp {det.disp_now:.1f}", (int(x1), y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
                cv2.putText(vis, f"Base {det.base_disp:.1f}", (int(x1), y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
                cv2.putText(vis, f"Ratio {det.ratio:.2f}/{det.ratio_th:.1f}", (int(x1), y0 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
                cv2.putText(vis, f"w:{det.weight:.2f} {det.reason}", (int(x1), y0 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)

        out.write(vis)
        cv2.imshow("Disp-Change Suspicion", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Saved disp_change_suspicion.mp4")

if __name__ == "__main__":
    main()