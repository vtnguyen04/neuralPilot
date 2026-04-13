import cv2
import json
import argparse
from pathlib import Path

class TimelineConfigurator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.segments = []

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame = 0
        self.start_marker = None
        self.paused = True

        cv2.namedWindow("Timeline Configurator")

    def run(self):
        print("\n--- Timeline Configurator (Enhanced) ---")
        print("Điều khiển:")
        print("  Space: Play/Pause (Để xem nhanh video)")
        print("  D / Mũi tên phải: Tiến 1 frame (Giữ Shift tiến 10)")
        print("  A / Mũi tên trái: Lùi 1 frame (Giữ Shift lùi 10)")
        print("  [: Đánh dấu START segment")
        print("  ]: Đánh dấu END segment")
        print("  0, 1, 2, 3: Gán Command ID cho đoạn đã chọn")
        print("  S: Lưu và Thoát | Q: Thoát không lưu")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.paused = True
                    self.current_frame = self.frame_count - 1
                else:
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret: break

            display_frame = frame.copy()

            # Hiển thị trạng thái
            mode = "PAUSED" if self.paused else "PLAYING"
            status = f"[{mode}] Frame: {self.current_frame}/{self.frame_count}"
            color = (0, 0, 255) if self.paused else (0, 255, 0)
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if self.start_marker is not None:
                cv2.putText(display_frame, f"MARKER START: {self.start_marker}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Liệt kê segment
            y_offset = 90
            for seg in self.segments[-3:]:
                txt = f"[{seg['start']}-{seg['end']}] -> CMD: {seg['command']}"
                cv2.putText(display_frame, txt, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            cv2.imshow("Timeline Configurator", display_frame)

            # Đợi phím (nếu đang chạy thì đợi ngắn, dừng thì đợi vô hạn)
            wait_time = int(1000 / self.fps) if not self.paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('['):
                self.start_marker = self.current_frame
                print(f"Bắt đầu đoạn tại frame: {self.start_marker}")
            elif key == ord(']'):
                if self.start_marker is not None:
                    print(f"Kết thúc đoạn tại frame: {self.current_frame}. Nhấn 0-3 để gán CMD.")
                else:
                    print("Nhấn '[' trước để chọn điểm đầu!")
            elif chr(key) in ['0', '1', '2', '3'] and self.start_marker is not None:
                cmd = int(chr(key))
                end_f = self.current_frame
                self.segments.append({
                    'start': min(self.start_marker, end_f),
                    'end': max(self.start_marker, end_f),
                    'command': cmd
                })
                print(f"Đã thêm: CMD {cmd} từ {self.start_marker} đến {end_f}")
                self.start_marker = None
            elif self.paused: # Các phím di chuyển chỉ dùng khi Pause cho chính xác
                if key == 83 or key == ord('d'):
                    self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
                elif key == 81 or key == ord('a'):
                    self.current_frame = max(self.current_frame - 1, 0)
                elif key == 84: # Shift + Right
                    self.current_frame = min(self.current_frame + 10, self.frame_count - 1)
                elif key == 82: # Shift + Left
                    self.current_frame = max(self.current_frame - 10, 0)

        self.cap.release()
        cv2.destroyAllWindows()

    def save(self):
        self.segments.sort(key=lambda x: x['start'])
        with open(self.output_path, 'w') as f:
            json.dump(self.segments, f, indent=4)
        print(f"Đã lưu timeline vào {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="timeline.json")
    args = parser.parse_args()
    TimelineConfigurator(args.source, args.output).run()
