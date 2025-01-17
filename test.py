from time import perf_counter
import torch
import numpy as np
from samurai.sam2.sam2.build_sam import build_sam2_video_predictor

video_folder_path = "./videos/bedroom"
cfg, ckpt = "./checkpoints/sam2_hiera_t.yaml", "./checkpoints/sam2.1_hiera_tiny.pt"
device = "cuda" # or "cpu"
predictor = build_sam2_video_predictor(cfg, ckpt, device)
inference_state = predictor.init_state(
    video_path=video_folder_path,
    async_loading_frames=False
)

predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[210, 350]], dtype=np.float32),
    labels=np.array([1], np.int32),
)

tprev = -1
for result in predictor.propagate_in_video(inference_state):
    # Do nothing with results, just report VRAM use
    if  (perf_counter() > tprev + 1.0) and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print("VRAM:", (total_bytes - free_bytes) // 1_000_000, "MB")
        tprev = perf_counter()
    pass