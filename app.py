import os
import cv2
from dotenv import load_dotenv

from src.animal_detector import AnimalDetector
from src.alert_notifier import AlertNotifier
from src.utils import put_label, Cooldown

def parse_alert_classes(val):
    if not val:
        return set()
    return {x.strip().lower() for x in val.split(",") if x.strip()}

def main():
    load_dotenv()

    video_source = os.getenv("VIDEO_SOURCE", "0")
    if video_source.isdigit():
        video_source = int(video_source)

    conf_thresh = float(os.getenv("CONF_THRESH", "0.5"))
    alert_classes = parse_alert_classes(os.getenv("ALERT_CLASSES", "cat,dog,bird,person"))
    cooldown_sec = int(os.getenv("ALERT_COOLDOWN_SEC", "120"))

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Check camera/index/path/URL.")

    detector = AnimalDetector(conf_thresh=conf_thresh)
    notifier = AlertNotifier()
    cooldown = Cooldown(cooldown_sec=cooldown_sec)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        boxes, labels, scores = detector.infer(frame)

        for (ymin, xmin, ymax, xmax), label, score in zip(boxes, labels, scores):
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 75), 2)
            put_label(frame, f"{label} {score:.2f}", x1, y1)

            if label.lower() in alert_classes and cooldown.ready(label.lower()):
                msg = f"[Backyard Monitor] Detected {label} with confidence {score:.2f}"
                if notifier.send(msg):
                    print(f"[ALERT] SMS sent: {msg}")
                else:
                    print(f"[ALERT] {msg}")

        cv2.imshow("Animals in Backyard — SSD MobileNetV2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
