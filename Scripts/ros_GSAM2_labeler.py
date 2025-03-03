# Need to use the GSAM env for this script
import sys
# REPO_PATH = "/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov2"
GSAM_path = '/home/mkhanum/Grounded-SAM-2'
sys.path.insert(1, GSAM_path)
# sys.path.append(REPO_PATH)

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import torch, time, pickle, joblib

import glob
import os
import cv2
import json
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert

from typing import Any, Tuple
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
# from segment_anything import build_sam, SamPredictor 
# from skimage.transform import resize
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
import open3d as o3d
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# import mmcv, urllib
# from mmcv.runner import load_checkpoint
# from mmseg.apis import init_segmentor, inference_segmentor
# import dinov2.eval.segmentation_m2f.models.segmentors

import groundingdino.datasets.transforms as T

# ==========================General Helper Functions==========================
SURROUND_U_STEP = 1.    #resolution
SURROUND_V_STEP = 1.
SURROUND_U_MIN, SURROUND_U_MAX = np.array([0,    360])/SURROUND_U_STEP  # horizontal of cylindrial projection
SURROUND_V_MIN, SURROUND_V_MAX = np.array([-90,   90])/SURROUND_V_STEP  # vertical   of cylindrial projection
# import ods
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ADE channel mapping to our channel definition, channel 0-6, ch 7 is bg/nothing
ade2our = np.array([7,3,3,7,0,4,7,0,4,4,6,4,0,5,6,2,4,7,4,4,4,5,4,4,4,4,3,4,3,0,6,4,4,3,4,4,4,4,4,4,4,4,4,3,4,4,4,6,4,3,4,4,7,0,1,0,4,4,4,2,1,4,3,4,4,4,4,4,4,6,4,4,4,4,4,4,4,5,4,4,3,5,4,7,5,3,4,7,4,3,4,5,0,4,4,0,4,1,4,4,4,3,0,5,5,4,4,7,4,4,4,4,4,4,3,4,4,5,4,4,4,4,1,4,4,4,4,5,5,4,4,4,4,4,4,4,4,4,4,4,4,7,4,4,4,4,7,4,4,3,4])
TEXT_PROMPT = "steps."

# IMG_PATH = "/home/mkhanum/datapipe/Frames/test_Color_1738272419265.39794921875000.png"
# DEPTH_PATH = "/home/mkhanum/datapipe/Frames/test_Depth_1738272419244.42651367187500.png"
# DEPTHCSV_PATH = "/home/mkhanum/datapipe/Frames/test_Depth_1738272421879.14868164062500.csv"

# IMG_PATH = "/home/mkhanum/datapipe/Frames/test_rgb.png"  # Update with actual path
# DEPTH_PATH = "/home/mkhanum/datapipe/Frames/test_depth.png"
# DEPTHCSV_PATH = "/home/mkhanum/datapipe/Frames/test_depth.csv"

IMG_PATH_LST = sorted(glob.glob("/home/mkhanum/smartbelt/aligned_vid4/color_*.png"))
DEPTHCSV_PATH_LST = sorted(glob.glob("/home/mkhanum/smartbelt/aligned_vid4/depth_*.png"))


SAM2_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "/home/mkhanum/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/mkhanum/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# OUTPUT_DIR = /home/mkhanum/datapipe/Segmentation

# SAM helper
def show_mask(mask, image, random_color=True, color=None):
    if color is None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


# ==========================Labeler Class==========================
class semantic_labeler:
    def __init__(self):
        self.device = DEVICE
        # self.bag_name = bag_name
        self.sam2_predictor = None
        self.grounding_model = None
        self.data_dict = None

        # Load GSAM2 environment components
        self.load_sam2()
        self.load_grounding_dino()
        # self.load_data_source()

    def load_sam2(self):
        """Load and initialize the SAM2 model."""
        print("Loading SAM2...")
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        print("SAM2 loaded successfully.")

    def load_grounding_dino(self):
        """Load and initialize the Grounding DINO model."""
        print("Loading Grounding DINO...")
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.device,
        )
        print("Grounding DINO loaded successfully.")

    def load_data_source(self):  # LOAD FRAME
        """Load an RGB image and a point cloud instead of a bag file."""
        self.video_frame_lst = []

        print(f"\nLoading RGB image from {IMG_PATH}")
        for img_path in IMG_PATH_LST:
            video_frame = cv2.imread(img_path)  # Load RGB image
            # if self.video_frame is None:
            #     raise ValueError(f"Failed to load image from {IMG_PATH}")
            # print("RGB image loaded successfully.")

            self.video_frame_lst.append(video_frame.copy())

        self.depth_values_lst = []
        for depth_path in DEPTHCSV_PATH_LST:
            self.depth_values_lst.append(np.loadtxt(depth_path, delimiter=",", skiprows=0))
            # print(self.depth_values.shape)

    def label_loaded_frames(self, img_lst = IMG_PATH_LST, text_prompt=TEXT_PROMPT, show_plot=False):
        """Label frames using GSAM2 components."""
        self.seg_frame_lst = []
        # frame = self.video_frame[:, :, ::-1]  # Convert to BGR

        for img_path in tqdm(img_lst):

            # tqdm.write(f"\nProcessing frame: {img_path}")
            
            frame, image = load_image(img_path)
            self.sam2_predictor.set_image(frame)
            
            # Predict bounding boxes and masks using Grounding DINO
            h, w, _ = frame.shape
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            
            if len(boxes) == 0:
                # If no labels are found, create a black mask
                seg_frame = np.zeros((h, w, 1))
            else:
                boxes = boxes * torch.Tensor([w, h, w, h])  # Scale boxes
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                
                # Predict masks with SAM2
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, _, _ = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                
                # Save segmented channels
                seg_frame = np.zeros((h, w, len(labels)))
                for ch_idx in range(len(labels)):
                    seg_frame[:, :, ch_idx] = masks[ch_idx]
            
            self.seg_frame_lst.append(seg_frame)

            save_fpath = '/home/mkhanum/datapipe/Output/vid4'
            for ch_idx in range(seg_frame.shape[2]):
                channel_image = seg_frame[:, :, ch_idx]
                plt.imsave(f"{save_fpath}/seg_frame_{os.path.basename(img_path).split('.')[0]}_channel_{ch_idx}.png", channel_image, cmap='gray')
            
            if show_plot:
                annotated_frame = sv.MaskAnnotator().annotate(
                    scene=frame, 
                    detections=sv.Detections(
                        xyxy=input_boxes if len(boxes) > 0 else np.array([]),
                        mask=masks.astype(bool) if len(boxes) > 0 else np.zeros((h, w, 1), dtype=bool),
                        class_id=np.arange(len(labels)) if len(boxes) > 0 else [],
                    )
                )
                plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
            
            # tqdm.write("Frame processed successfully.")


    # =====================
    # Segmented Frame to PC (local) to PC (global)
    # =====================
    def get_3d_points_from_mask(self, depth_image, segmentation_mask):
        pc_frame = depth_image*10
        fx = 431.0087890625
        fy = 431.0087890625
        ppx = 429.328704833984
        ppy = 242.162155151367

        points_3d = []
        height, width = depth_image.shape
        scale_x = depth_image.shape[1] / segmentation_mask.shape[1]  # width scaling factor
        scale_y = depth_image.shape[0] / segmentation_mask.shape[0]  # height scaling factor

        for y_color in range(self.video_frame.shape[0]):
            for x_color in range(self.video_frame.shape[1]):
                if segmentation_mask[y_color, x_color]:  # Check if the pixel is part of the segmentation mask
                # if segmentation_mask[v, u] > 0:  # If the pixel is part of the region of interest
                    # depth = depth_image[v, u]
                    x_depth = int(x_color * scale_x)
                    y_depth = int(y_color * scale_y)
                    

                    depth = depth_image[y_depth, x_depth]
                    
                    if depth > 0:  # Ignore zero depth (invalid points)
                        # Convert depth to 3D coordinates
                        x = (u - ppx) * depth / fx
                        y = (v - ppy) * depth / fy
                        z = depth
                        
                        # Optionally, get the color from the RGB image
                        r, g, b = rgb_image[v, u]
                        
                        # Append the 3D point with the RGB values (optional)
                        points_3d.append([x, y, z, r, g, b])

                        # points_3d.append([x,y,z])

        return np.array(points_3d)

    def apply_segmentation_to_frames(self):
        """
        Applies the segmented mask to the video and point cloud frames.
        Only keeps the RGB and depth values associated with the segmented part.
        
        Parameters:
        - seg_frame: A 3D mask (height, width, num_classes) indicating the segmented parts.
        - video_frame: The RGB video frame (height, width, 3).
        - pc_frame: The point cloud frame (height, width, 3), corresponding to depth data.
        
        Returns:
        - masked_video_frame: The RGB video frame with only the segmented parts.
        - masked_pc_frame: The point cloud with only the segmented points.
        """
        
        # Initialize masked frames
        self.masked_pc_frame_lst = []
        # self.masked_video_frame_lst = []
        # self.masked_depth_frame_lst = []  # Initialize depth frame list
        # pc_frame = depth_image*10
        fx = 431.0087890625
        fy = 431.0087890625
        ppx = 429.328704833984
        ppy = 242.162155151367

        points_3d = []

        # Get scaling factors
        scale_x = self.depth_values_lst[0].shape[1] / self.video_frame_lst[0].shape[1]  # width scaling factor
        scale_y = self.depth_values_lst[0].shape[0] / self.video_frame_lst[0].shape[0]  # height scaling factor

        # masked_video_frame_lst = []
        # masked_depth_frame_lst = []

        # masked_video_frame = np.zeros_like(self.video_frame)
        # masked_depth_frame = np.zeros_like(self.depth_frame)  # Initialize depth frame

        # seg_frame = self.seg_frame_lst

        min_depth = 2000

        for i, seg_frame in enumerate(self.seg_frame_lst):
            # masked_video_frame = np.zeros_like(self.video_frame)
            # masked_depth_frame = np.zeros_like(self.depth_frame)  # Initialize depth frame
            for ch_idx in range(1): #seg_frame.shape[2]
                mask = seg_frame[:, :, ch_idx] > 0  # Get the mask for class ch_idx
                
                # For each pixel in the color image, map it to the depth image coordinates
                for y_color in range(self.video_frame.shape[0]):
                    for x_color in range(self.video_frame.shape[1]):
                        if mask[y_color, x_color]:  # Check if the pixel is part of the segmentation mask
                            # Apply the mask to the RGB frame: only keep values where mask is True
                            # video_frame = self.video_frame_lst[i]
                            depth_values = self.depth_values_lst[i]
                            # masked_video_frame[y_color, x_color] = self.video_frame[y_color, x_color]
                            # Map the color image coordinates to the depth image coordinates

                            x_depth = int(x_color * scale_x)
                            y_depth = int(y_color * scale_y)
                            
                            # Apply the mask to the depth frame: only keep depth values where mask is True
                            # masked_depth_frame[y_depth, x_depth] = self.depth_frame[y_depth, x_depth]

                            depth = depth_values[y_depth, x_depth]
                        
                            if depth > 0:  # Ignore zero depth (invalid points)
                                # Convert depth to 3D coordinates
                                x = (x_color - ppx) * depth / fx
                                y = (y_color - ppy) * depth / fy
                                z = depth
                                
                                # Optionally, get the color from the RGB image
                                # r, g, b = rgb_image[v, u]
                                # points_3d.append([x, y, z, r, g, b])

                                points_3d.append([x,y,z])

                                if depth < min_depth:
                                    min_depth = depth
            self.masked_pc_frame.append(points_3d)
                
                # print("HERE")

        # self.masked_video_frame = masked_video_frame
        # self.masked_depth_frame = masked_depth_frame
        # self.masked_pc_frame = points_3d

        # print(min(points_3d))

        # print(min_depth)

        # self.masked_pc_frame = self.get_3d_points_from_mask(self.video_frame, self.depth_values, self.seg_frame_lst)

    # def apply_segmentation_to_depth(self):
        

    # D455
    def depth_to_pc(self, depth_frame, seg_frame, video_frame):
        pc_frame = depth_frame*10
        depth_fx = 431.0087890625
        depth_fy = 431.0087890625
        depth_ppx = 429.328704833984
        depth_ppy = 242.162155151367

        @nb.jit(nopython=True)
        def func_nb():
            values = []
            for u in range(848):
                for v in range(480):
                    if pc_frame[v, u] > 0.3:
                        # depth data
                        depth = pc_frame[v, u, 0]
                        x = (u - depth_ppx) * depth / depth_fx
                        y = (v - depth_ppy) * depth / depth_fy
                        z = depth

                        # color data
                        r,g,b = video_frame[v,u,:]

                        # segmentation data
                        c1,c2,c3,c4,c5,c6,c7 = seg_frame[v,u,:]
                        # print(type(x), type(c1), type(r))

                        # save to dictionary
                        # pt_dict[(v,u)] = [x,y,z, r,g,b, c1,c2,c3,c4,c5,c6,c7]
                        # keys.append((v,u))
                        values.append([x,y,z, r,g,b, c1,c2,c3,c4,c5,c6,c7])
            return values
        
        values = func_nb()
        # return dict(zip(keys, values))
        return values

    def depth_to_pcD435(self, depth_frame, seg_frame, video_frame, hfov=60, vfov=45):
        depth_frame = depth_frame[:,22:,0]
        depth_frame = resize(depth_frame, (458, 590), order=0, preserve_range=True)
        pc_frame = np.zeros((480, 640))
        pc_frame[2:460, 22:612] = depth_frame*10

        # tan(ver_ang_rad)*z = y
        # tan(hor_ang_rad)*z = x
        # save points as a dictionary
        @nb.jit(nopython=True)
        def func_nb():
            values = []
            for u in range(640):
                for v in range(480):
                    if pc_frame[v, u] > 0.3:
                        # Point data
                        hor_ang_rad = np.deg2rad((u-320)/640*hfov)
                        ver_ang_rad = np.deg2rad((v-240)/480*vfov)

                        z = pc_frame[v, u]
                        x = np.tan(hor_ang_rad)*z
                        y = np.tan(ver_ang_rad)*z

                        # color data
                        r,g,b = video_frame[v,u,:]

                        # segmentation data
                        c1,c2,c3,c4,c5,c6,c7 = seg_frame[v,u,:]

                        # save to dictionary
                        # pt_dict[(v,u)] = [x,y,z, r,g,b, c1,c2,c3,c4,c5,c6,c7]
                        # keys.append((v,u))
                        values.append([x,y,z, r,g,b, c1,c2,c3,c4,c5,c6,c7])
            return values
        
        values = func_nb()
        # return dict(zip(keys, values))
        return values

    # =====================
    # General Helper Functions
    # =====================

    def find_nearest_pos(self,t):
        idx = np.argmin(abs(self.data_dict['data_array'][:,0] - t))
        return self.data_dict['data_array'][idx,1:4].copy(), self.data_dict['data_array'][idx,4:8].copy()

    def pc_video_idxs(self):
        # pc_t    = np.array(self.data_dict['pc_t'])
        # video_t = np.array(self.data_dict['video_t'])
        # idxs    = []

        # for i in range(len(pc_t)):
        #     curr_t   = pc_t[i]
        #     curr_idx = np.argmin(abs(video_t-curr_t))
        #     idxs.append(curr_idx)

        # return np.array(idxs)

        return np.array([0])

    def intensity_labeling(self, seg=False):
        if seg:
            seg_frame = np.array(self.data_dict['seg_frame'])
            # if np.max(np.sum(seg_frame, axis=-1)) > 1.1:
            #     print('ERROR: channels are not one hot')
            #     return 0
            
            seg_frame_int = np.argmax(seg_frame, axis=-1)
            seg_frame_int[np.sum(seg_frame, axis=-1)==0] = 7 # no label channel
            self.data_dict['seg_frame']  = seg_frame_int

        # repeat for panorama
        # pano_frame_rgbd = self.data_dict['pano_frame'][:,:,:,:4]
        pano_frame_seg = self.data_dict['pano_frame'][:,:,:,4:]
        # if np.max(np.sum(pano_frame, axis=-1)) > 1.1:
        #     print('ERROR: channels are not one hot')
        #     return 0

        pano_frame_int = np.argmax(pano_frame_seg, axis=-1)
        pano_frame_int[np.sum(pano_frame_seg, axis=-1)==0] = 7 # no label channel
        self.data_dict['pano_frame'] = np.concatenate([self.data_dict['pano_frame'][:,:,:,:4], pano_frame_int[..., np.newaxis]], axis=-1, dtype=np.float32)

    def save_training_set(self):
        self.data_dict['prompts'] = prompts
        
        # save as new dataset
        print('Saving to disk...')
        t_start = time.time()

        save_fpath = '/home/mkhanum/datapipe/Training_sets/eDS20HZVZS_' + self.bag_name[:-4]
        # outfile = open(save_fpath,'wb')
        # pickle.dump(self.data_dict, outfile)
        # outfile.close()
        joblib.dump(self.data_dict, save_fpath, compress=('lz4', 1))

        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))
    
    def save_frames(self):
        save_fpath = '/home/mkhanum/datapipe/Output/Frame'

        # Ensure the save directory exists
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath)

        # # Iterate over each channel (class) and save it
        # for ch_idx in range(self.seg_frame_lst.shape[2]):
        #     # Extract each channel (2D array)
        #     channel_image = self.seg_frame_lst[:, :, ch_idx]
            
        #     # Save each channel as a separate PNG
        #     plt.imsave(f"{save_fpath}/seg_frame_lst_channel_{ch_idx}.png", channel_image, cmap='gray')
        # Save each segmented frame
            for ch_idx in range(seg_frame.shape[2]):
                channel_image = seg_frame[:, :, ch_idx]
                plt.imsave(f"{save_fpath}/seg_frame_{os.path.basename(img_path).split('.')[0]}_channel_{ch_idx}.png", channel_image, cmap='gray')
        

        # # Save masked_video_frame
        # if self.masked_video_frame is not None:
        #     plt.imsave(f"{save_fpath}/masked_video_frame.png", self.masked_video_frame.astype(np.uint8))

        # # Save masked_depth_frame
        # if self.masked_depth_frame is not None:
        #     plt.imsave(f"{save_fpath}/masked_depth_frame.png", self.masked_depth_frame.astype(np.uint8))

        # if self.masked_pc_frame is not None and len(self.masked_pc_frame) > 0:
        #     # Convert to Open3D PointCloud
        #     pc = o3d.geometry.PointCloud()
        #     pc.points = o3d.utility.Vector3dVector(self.masked_pc_frame)

        #     # Save as PLY file
        #     o3d.io.write_point_cloud(f"{save_fpath}/masked_pc_frame.ply", pc)

        print("Frames saved successfully.")



if __name__ == '__main__':
    # Prompt, how confident is the box, how closely should it match text, rgba
    # prompts = [["ground, sidewalk",0.2,0.28,[0.1, 0.1, 0.1, 0.7]], # dark
    #            ["stairs",0.4,0.4,[0.7, 0.7, 0.7, 0.7]], # gray ish
    #     #    ["ramp",0.3,0.25,[0.5, 0.5, 0.5, 0.7]],   # dark ish
    #            ["door, wood door, steel door, glass door, elevator door",0.55,0.5,[0.0, 0.7, 0.0, 0.7]],   # green
    #            ["wall, pillar",0.47,0.3,[0.7, 0.0, 0.0, 0.7]],   # red
    #            ["bin, chair, bench, desk, plants, curb, bushes, pole, tree",0.55,0.5,[0.0, 0.0, 0.7, 0.7]],
    #            ["people, person, pedestrian", 0.5, 0.5, [0.0, 0.0, 0.7, 0.7]]] # blue

    # ==============Load model and image to label================
    # Load data

    t1 = time.time()
    labeler = semantic_labeler()
    t2 = time.time()
    print("Loading took: %2.2f seconds"%(t2-t1))