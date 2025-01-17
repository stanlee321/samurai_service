import os
import os.path as osp
import uuid
import numpy as np
import cv2
import torch
import gc
import json
import sys
import warnings
sys.path.append("./samurai/sam2/sam2")
from typing import List
from libs.queues import KafkaHandler
from samurai.sam2.sam2.build_sam import build_sam2_video_predictor

SERVER_IP = "192.168.1.15"
API_BASE_URL = f"http://{SERVER_IP}:8003"

minio_key = "c3aFDmKuGhPCSxkpRDGf"
minio_secret = "MLYz9tZI3h4xZAwBl8llyEtX6R07YcMuRdSYPIcx"
minio_url = f"{SERVER_IP}:9000"

BUCKET_NAME = "my-bucket"
TOPIC_INPUT = "video-general-sam"
TOPIC_OUTPUT = "video-fine-detections"

brokers = [f'{SERVER_IP}:9092']

WORKING_FOLDER = "./tmp"



class VideoProcessor:
    def __init__(self, device=None, brokers: List[str] = None):
        self.color = [(255, 0, 0)]
        self.device = device or self._get_optimal_device()
        print(f"Using device: {self.device}")
        
        
        self.kafka_handler = KafkaHandler(bootstrap_servers=brokers)

        
    def _get_optimal_device(self):
        """Get the optimal available device with proper configuration"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
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

    def load_coords(self, coords: List[str]) -> dict:
        prompts = {}
        for fid, line in enumerate(coords):
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

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()

        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video")

        height, width = loaded_frames[0].shape[:2]
        return loaded_frames, height, width, frame_rate

    def process_video(self, model_path, video_path, coords, video_output_path) -> str:
        """
        Process video and return the path to the output video
        """
        # Initialize model and load data
        model_cfg = self.determine_model_cfg(model_path)
        predictor = build_sam2_video_predictor(model_cfg, model_path, device=self.device)
        frames_or_path = self.prepare_frames_or_path(video_path)
        prompts = self.load_coords(coords)

        # Load video frames
        loaded_frames, height, width, frame_rate = self.load_video_frames(video_path)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (width, height))

        # Process frames
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    self._process_frames(predictor, frames_or_path, prompts, loaded_frames, height, width, out)
            else:
                # For MPS and CPU, run without autocast
                self._process_frames(predictor, frames_or_path, prompts, loaded_frames, height, width, out)

        # Cleanup
        out.release()
        self._cleanup(predictor)
        
        return video_output_path

    def _process_frames(self, predictor, frames_or_path, prompts, loaded_frames, height, width, out):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
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

            self._visualize_frame(loaded_frames[frame_idx], mask_to_vis, bbox_to_vis, height, width, out)

    def _visualize_frame(self, img, mask_to_vis, bbox_to_vis, height, width, out):
        for obj_id, mask in mask_to_vis.items():
            mask_img = np.zeros((height, width, 3), np.uint8)
            mask_img[mask] = self.color[(obj_id + 1) % len(self.color)]
            img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

        for obj_id, bbox in bbox_to_vis.items():
            cv2.rectangle(img, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                         self.color[obj_id % len(self.color)], 2)

        out.write(img)

    def _cleanup(self, predictor):
        del predictor
        gc.collect()
        torch.clear_autocast_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def generate_uuid(self):
        return str(uuid.uuid4())
    
    def run(self, offset: str = "latest"):
        group_id = 'video-samurai-' + self.generate_uuid()
        consumer = self.kafka_handler.create_consumer(TOPIC_INPUT,
                                                      group_id=group_id,
                                                      auto_offset_reset=offset)

        for message in consumer:
            print(f"Consumed message: {message.value}")
            message_value = json.loads(message.value)
            
            print("...")
            print(message_value)
            
            model_path = message_value["model_path"]
            video_path = message_value["video_path"]
            coords = message_value["coords"]
            video_output_path = message_value["video_output_path"]
            
            model_path = "./checkpoints/sam2.1_hiera_base_plus.pt"
            video_path = "./samurai/data/video.mp4"
            video_output_path = "./output.mp4"
            coords = ["575,90,65,120"]
    
            self.process_video(model_path, video_path, coords, video_output_path)

def main():
    # Initialize processor
    processor = VideoProcessor()
    
    # Define paths
    # model_path = "./checkpoints/sam2.1_hiera_base_plus.pt"
    # video_path = "./videos/bedroom"
    # video_output_path = "./output.mp4"
    # coords = ["575,90,65,120"]
    # Process video
    processor.run()

if __name__ == "__main__":
    main()
