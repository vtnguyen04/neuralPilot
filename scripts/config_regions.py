import cv2
import yaml
import argparse
from pathlib import Path

class RegionConfigurator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.regions = []
        self.current_region = None
        self.drawing = False
        self.ix, self.iy = -1, -1

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        cv2.namedWindow("Region Configurator")
        cv2.setMouseCallback("Region Configurator", self.draw_region)

    def draw_region(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_region = (self.ix, self.iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_region = (self.ix, self.iy, x, y)
            print(f"Region selected: {self.current_region}")
            cmd = input("Enter Command ID (0:FOLLOW, 1:LEFT, 2:RIGHT, 3:STRAIGHT): ")
            try:
                cmd = int(cmd)
                start_f = int(input(f"Start frame (current: {self.current_frame}): ") or self.current_frame)
                end_f = int(input(f"End frame (max: {self.frame_count}): ") or self.frame_count)
                self.regions.append({
                    'box': [self.ix, self.iy, x, y],
                    'command': cmd,
                    'start': start_f,
                    'end': end_f
                })
                print(f"Added region: CMD={cmd}, Frames=[{start_f}, {end_f}]")
            except ValueError:
                print("Invalid input. Region discarded.")
            self.current_region = None

    def run(self):
        print("\n--- Region Configurator ---")
        print("Controls:")
        print("  Space: Play/Pause")
        print("  D: Next frame")
        print("  A: Previous frame")
        print("  S: Save and Exit")
        print("  Q: Quit without saving")
        print("  L: Click and drag to draw a region")

        paused = True
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    continue
                self.current_frame += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()

            display_frame = frame.copy()

            # Draw existing regions
            for r in self.regions:
                if r['start'] <= self.current_frame <= r['end']:
                    x1, y1, x2, y2 = r['box']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"CMD: {r['command']}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw current drawing region
            if self.current_region:
                x1, y1, x2, y2 = self.current_region
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(display_frame, f"Frame: {self.current_frame}/{self.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Region Configurator", display_frame)
            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)

        self.cap.release()
        cv2.destroyAllWindows()

    def save(self):
        data = {
            'video': str(self.video_path),
            'regions': self.regions
        }
        with open(self.output_path, 'w') as f:
            yaml.dump(data, f)
        print(f"Regions saved to {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="regions.yaml", help="Output YAML file")
    args = parser.parse_args()

    configurator = RegionConfigurator(args.source, args.output)
    configurator.run()
