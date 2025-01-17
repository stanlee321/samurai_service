import os
import warnings
import numpy as np
import torch
import cv2
import gc
import argparse
import os.path as osp
from PIL import Image
import sys
sys.path.append("./samurai/sam2/sam2")
from samurai.sam2.sam2.build_sam import build_sam2_video_predictor

class VideoProcessor:
    def __init__(self, device):
        self.device = device
        self.color = [(255, 0, 0)]
        
    def load_txt(self, gt_path):
        with open(gt_path, 'r') as f:
            gt = f.readlines()
        prompts = {}
        for fid, line in enumerate(gt):
            x, y, w, h = map(float, line.split(','))
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
        return prompts

    def determine_model_cfg(self, model_path):
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            raise ValueError("Unknown model size in path!")

    def prepare_frames_or_path(self, video_path):
        if video_path.endswith(".mp4") or osp.isdir(video_path):
            return video_path
        else:
            raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

    def process_video(self, args):
        model_cfg = self.determine_model_cfg(args.model_path)
        predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=self.device)
        frames_or_path = self.prepare_frames_or_path(args.video_path)
        prompts = self.load_txt(args.txt_path)

        frame_rate = 30
        if args.save_to_video:
            loaded_frames, height, width, frame_rate = self.load_video_frames(args.video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

        with torch.inference_mode():
            if self.device.type == "cuda":
                # CUDA supports autocast
                with torch.autocast("cuda", dtype=torch.float16):
                    self.process_frames(predictor, frames_or_path, prompts, loaded_frames, out, args)
            else:
                # For MPS and CPU, run without autocast
                self.process_frames(predictor, frames_or_path, prompts, loaded_frames, out, args)

        if args.save_to_video:
            out.release()

        self.cleanup(predictor)

    def load_video_frames(self, video_path):
        if osp.isdir(video_path):
            frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) 
                           if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
            frame_rate = 30
        else:
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            if not loaded_frames:
                raise ValueError("No frames were loaded from the video.")
            height, width = loaded_frames[0].shape[:2]

        return loaded_frames, height, width, frame_rate

    def process_frames(self, predictor, frames_or_path, prompts, loaded_frames, out, args):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis, bbox_to_vis = self.process_masks(object_ids, masks)
            
            if args.save_to_video:
                self.visualize_frame(loaded_frames[frame_idx], mask_to_vis, bbox_to_vis, out)

    def process_masks(self, object_ids, masks):
        mask_to_vis = {}
        bbox_to_vis = {}
        
        for obj_id, mask in zip(object_ids, masks):
            mask = mask[0].cpu().numpy()
            mask = mask > 0.0
            non_zero_indices = np.argwhere(mask)
            
            if len(non_zero_indices) == 0:
                bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
            bbox_to_vis[obj_id] = bbox
            mask_to_vis[obj_id] = mask
            
        return mask_to_vis, bbox_to_vis

    def visualize_frame(self, img, mask_to_vis, bbox_to_vis, out):
        height, width = img.shape[:2]
        for obj_id, mask in mask_to_vis.items():
            mask_img = np.zeros((height, width, 3), np.uint8)
            mask_img[mask] = self.color[(obj_id + 1) % len(self.color)]
            img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

        for obj_id, bbox in bbox_to_vis.items():
            cv2.rectangle(img, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                         self.color[obj_id % len(self.color)], 2)

        out.write(img)

    def cleanup(self, predictor):
        del predictor
        gc.collect()
        torch.clear_autocast_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

def get_optimal_device():
    """Get the optimal available device with proper configuration"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        warnings.warn(
            "\nUsing MPS device. Note: Some operations may fall back to CPU.",
            RuntimeWarning
        )
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="./checkpoints/sam2.1_hiera_base_plus.pt", 
                        help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", 
                        help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()

    device = get_optimal_device()
    print(f"Using device: {device}")

    processor = VideoProcessor(device)
    processor.process_video(args)

if __name__ == "__main__":
    main()