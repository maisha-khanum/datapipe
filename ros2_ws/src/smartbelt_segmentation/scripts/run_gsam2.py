import sys
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

TEXT_PROMPT = "steps."
SAM2_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "/home/mkhanum/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def run_gsam2(image_path, mask_path):
    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return False
    image = image[:, :, ::-1]  # Convert to RGB

    frame = np.asarray(image) # TODO, may need to replace with load_image()
    image,_ = transform(image)

    # Load Models
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE)

    # Run Segmentation
    sam2_predictor.set_image(image)
    h, w, _ = frame.shape
    boxes, _, _ = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Predict Masks
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Convert Masks to Single Channel
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    for ch_idx in range(masks.shape[0]):
        seg_mask[masks[ch_idx] > 0] = 255

    # Save Mask
    cv2.imwrite(mask_path, seg_mask)
    print("Mask saved successfully.")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_gsam2.py <image_path> <mask_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    if not run_gsam2(image_path, mask_path):
        sys.exit(1)
