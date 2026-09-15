"""Turn a directory of rendered frames into an mp4 and a (downscaled) animated gif."""
import sys
from pathlib import Path

import cv2
from PIL import Image

frame_dir, out_stem, fps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
frames = sorted(frame_dir.glob('*.png'))
if not frames:
    sys.exit(f'no frames in {frame_dir}')

height, width = cv2.imread(str(frames[0])).shape[:2]
writer = cv2.VideoWriter(str(out_stem.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for frame in frames:
    writer.write(cv2.imread(str(frame)))
writer.release()

gif_frames = [Image.open(f).convert('RGB').resize((width // 2, height // 2), Image.LANCZOS) for f in frames]
gif_frames[0].save(
    out_stem.with_suffix('.gif'), save_all=True, append_images=gif_frames[1:],
    duration=int(1000 / fps), loop=0, optimize=True,
)
print(f'{len(frames)} frames -> {out_stem.with_suffix(".mp4")} ({width}x{height}) + .gif')
