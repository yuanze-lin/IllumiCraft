import cv2
import numpy as np

video_path = "input_video.mp4"
mask_path = "mask_video.mp4" # foreground pixels are (255, 255, 255) and background pixels are (0, 0, 0)
out_path = "foreground_video.mp4"

cap_v = cv2.VideoCapture(video_path)
cap_m = cv2.VideoCapture(mask_path)

fps = cap_v.get(cv2.CAP_PROP_FPS)
w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

# Background color: #888b88
background = np.full((h, w, 3), (136, 139, 136), dtype=np.uint8)

while True:
    ret_v, frame = cap_v.read()
    ret_m, mask_frame = cap_m.read()

    if not ret_v or not ret_m:
        break

    # Convert mask video to binary foreground mask
    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
    mask = mask[..., None]

    # Foreground + gray background
    output = frame * mask + background * (1 - mask)

    writer.write(output.astype(np.uint8))

cap_v.release()
cap_m.release()
writer.release()

print(f"Saved to {out_path}")
