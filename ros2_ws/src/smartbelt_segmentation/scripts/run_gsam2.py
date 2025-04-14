import sys
# REPO_PATH = "/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov2"
GSAM_path = '/home/mkhanum/Grounded-SAM-2'
sys.path.insert(1, GSAM_path)

from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert


app = Flask(__name__)

# Load models once when the server starts
TEXT_PROMPT = "steps."
SAM2_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "/home/mkhanum/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE)

@app.route('/run_gsam2', methods=['POST'])
def run_gsam2():
    data = request.json
    image_path = data['image_path']
    mask_path = data['mask_path']

    frame, image = load_image(image_path)
    sam2_predictor.set_image(frame)
    h, w, _ = frame.shape
    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=0.35,
        text_threshold=0.25
    )

    if len(boxes) == 0:
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(mask_path, seg_mask)
        return jsonify(success=True)

    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )

    seg_mask = np.zeros((h, w), dtype=np.uint8)
    seg_mask[masks[0] > 0] = 255
    cv2.imwrite(mask_path, seg_mask)
    
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
