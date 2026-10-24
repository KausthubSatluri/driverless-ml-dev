import argparse
import os
import sys
import time

import cv2
import psutil
import torch
from PIL import Image

# Each frame is labelled on its own: no tracking, so nothing builds up over a long video.
# Usage:
#   python label_frames.py path/to/video.mp4
#   python label_frames.py path/to/video.mp4 --output out.mp4 --prompts "blue cone" "yellow cone"
parser = argparse.ArgumentParser(description="Draw SAM 3 boxes on every frame of a video.")
parser.add_argument("video", help="path to the input video")
parser.add_argument("--output", help="where to save the labelled video (default: <video>_labelled.mp4 next to it)")
parser.add_argument("--prompts", nargs="+", default=["blue cone", "yellow cone", "orange cone"],
                    help="text prompts to look for, one box color each")
args = parser.parse_args()

path = args.video
output = args.output or os.path.splitext(path)[0] + "_labelled.mp4"
items = args.prompts

video = cv2.VideoCapture(path)
if not video.isOpened():
    sys.exit(f"Could not open video: {path}")

from model import processor  # loaded after the checks above, since it takes a while

# Settings from SAM's own image example: faster maths on recent GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # BGR
text_size = width / 1500  # bigger videos get bigger text
thickness = max(2, width // 500)

frames_with_box = {item: 0 for item in items}
start = time.time()
frame_index = 0
while True:
    ok, image = video.read()  # read the next frame
    if not ok:
        break  # no frames left

    # SAM wants an RGB picture; OpenCV gives BGR
    picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(picture)  # look at the frame once, reuse it for every word
        for item_index, item in enumerate(items):
            color = colors[item_index % len(colors)]  # one color per word
            state = processor.set_text_prompt(prompt=item, state=state)
            boxes = state["boxes"].float().cpu().numpy()  # (x0, y0, x1, y1) in pixels
            scores = state["scores"].float().cpu().numpy()
            if len(boxes) > 0:
                frames_with_box[item] += 1

            for (x0, y0, x1, y1), score in zip(boxes, scores):
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
                cv2.putText(image, f"{item} {score:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)

    writer.write(image)  # add the labelled frame to the new video
    frame_index += 1

    if frame_index % 100 == 0:
        speed = frame_index / (time.time() - start)
        gpu_peak = torch.cuda.max_memory_allocated() / 1e9
        ram_used = psutil.Process().memory_info().rss / 1e9
        print(f"frame {frame_index}/{total_frames}: {speed:.1f} frames/s, peak GPU {gpu_peak:.1f} GB, RAM {ram_used:.1f} GB")

video.release()
writer.release()
for item in items:
    print(f"{item}: {frames_with_box[item]} of {frame_index} frames with at least one box")
print(f"Saved {output} ({frame_index} frames, {time.time() - start:.0f} s)")
