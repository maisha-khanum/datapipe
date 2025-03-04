# Need to use the GSAM env for this script
import sys
# REPO_PATH = "/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov2"
GSAM_path = '/home/mkhanum/Grounded-SAM-2'
sys.path.insert(1, GSAM_path)
# sys.path.append(REPO_PATH)

sys.path.append("/home/mkhanum/Grounded-SAM-2/grounding_dino")

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import torch, time, pickle, joblib

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

# prompts = [["ground, sidewalk",0.2,0.28,[0.1, 0.1, 0.1, 0.7]], # dark
#            ["stairs",0.4,0.4,[0.7, 0.7, 0.7, 0.7]], # gray ish
#            ["door, wood door, steel door, glass door, elevator door",0.55,0.5,[0.0, 0.7, 0.0, 0.7]],   # green
#            ["wall, pillar",0.47,0.3,[0.7, 0.0, 0.0, 0.7]],   # red
#            ["bin, chair, bench, desk, plants, curb, bushes, pole, tree",0.55,0.5,[0.0, 0.0, 0.7, 0.7]],
#            ["people, person, pedestrian", 0.5, 0.5, [0.05, 0.7, 0.7, 0.7]],
#            ["grass, field, sand, hill, earth, dirt", 0.5, 0.5, [0.5,0.5,0.5,0.7]]]

TEXT_PROMPT = "steps."

IMG_PATH = "/home/mkhanum/datapipe/Frames/test_Color_1738272419265.39794921875000.png"
DEPTH_PATH = "/home/mkhanum/datapipe/Frames/test_Depth_1738272419244.42651367187500.png"
DEPTHCSV_PATH = "/home/mkhanum/datapipe/Frames/test_Depth_1738272421879.14868164062500.csv"

# IMG_PATH = "/home/mkhanum/datapipe/Frames/test_rgb.png"  # Update with actual path
# DEPTH_PATH = "/home/mkhanum/datapipe/Frames/test_depth.png"
# DEPTHCSV_PATH = "/home/mkhanum/datapipe/Frames/test_depth.csv"


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


def uv2xyz(u, v, depth):
    # D455 depth intrinsics
    depth_fx = 431.0087890625
    depth_fy = 431.0087890625
    depth_ppx = 429.328704833984
    depth_ppy = 242.162155151367

    x = (u - depth_ppx) * depth / depth_fx
    y = (v - depth_ppy) * depth / depth_fy
    z = depth
    return x,y,z


def generate_pano(pc_frame_glob, curr_pos,r,sample_ratio,view_dist=10.0,filter=False):  
      
    total_pt_array = np.concatenate([pc_frame[0] for pc_frame in pc_frame_glob],axis=0)
    remain_idx = np.random.choice(total_pt_array.shape[0], int(total_pt_array.shape[0]*sample_ratio), replace=False)
    total_pt_array = total_pt_array[remain_idx]

    total_pt_array[:,:3] = total_pt_array[:,:3]-curr_pos
    total_pt_array[:,:3] = total_pt_array[:,:3].dot(r) # rotate to user heading

    x = total_pt_array[:,0]
    y = total_pt_array[:,1]
    z = total_pt_array[:,2]

    dxy = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin, this is projected dist onto xy plane
    dist = np.sqrt(dxy**2 + z**2) # xyz distance

    ##### Filter d based on distance #####
    if filter:
        before = len(x)
        x = x[dist<=view_dist]
        y = y[dist<=view_dist]
        z = z[dist<=view_dist]
        dxy = dxy[dist<=view_dist]
        total_pt_array = total_pt_array[dist<=view_dist,:]
        dist = dist[dist<=view_dist]
        after = len(x)
        print('before: ',before,' after: ',after , 'ratio: ', after/before)

    u,v = lidar_to_surround_coords(x,y,z,dxy)

    width  = int(SURROUND_U_MAX - SURROUND_U_MIN)
    height = int(SURROUND_V_MAX - SURROUND_V_MIN)

    # surround_img = np.zeros((height, width), dtype=np.uint8)+255
    surround_img = np.zeros((height, width,11)) # rgbd c1-c7
    surround_img[:,:,3] = 255 # d=255 if unknown

    @nb.njit # OPTIMIZED
    def proj_and_clip(tmp, u,v,d, total_pt_array): # OPTIMIZED
        for i in range(len(u)):
            if tmp[v[i],u[i],3] > d[i]:
                _, _, _, r,g,b, c1,c2,c3,c4,c5,c6,c7 = total_pt_array[i,:]
                tmp[v[i],u[i],:] = np.array([r,g,b, d[i], c1,c2,c3,c4,c5,c6,c7])

    dist255 = normalise_to_255(np.clip(dist, 0.0, 10.0))
    proj_and_clip(surround_img, u,v,dist255, total_pt_array)

    return surround_img

def lidar_to_surround_coords(x, y, z, dxy):
    u =   np.arctan2(x, y)/np.pi*180 /SURROUND_U_STEP
    v = - np.arctan2(z, dxy)/np.pi*180 /SURROUND_V_STEP
    u = (u + 90 + 360)%360

    u = np.floor(u)
    v = np.floor(v)
    u = (u - SURROUND_U_MIN).astype(np.uint16)
    v = (v - SURROUND_V_MIN).astype(np.uint16)
    return u,v

@nb.njit # OPTIMIZED
def normalise_to_255(a):
    # return (((a - min(a)) / float((max(a) - min(a))*0+10)) * 255.0).astype(np.uint8)
    return (a/10.0*255.0).astype(np.uint8)

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
# ==========================Labeler Class==========================
class semantic_labeler:
    def __init__(self, bag_name):
        self.device = DEVICE
        self.bag_name = bag_name
        self.sam2_predictor = None
        self.grounding_model = None
        self.data_dict = None

        # Load GSAM2 environment components
        self.load_sam2()
        self.load_grounding_dino()
        self.load_data_source()

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

        print(f"\nLoading RGB image from {IMG_PATH}")
        self.video_frame = cv2.imread(IMG_PATH)  # Load RGB image
        if self.video_frame is None:
            raise ValueError(f"Failed to load image from {IMG_PATH}")
        print("RGB image loaded successfully.")

        self.video_frame = self.video_frame.copy()

        # print(f"\nLoading Point Cloud from {PC_PATH}")
        # pcd = o3d.io.read_point_cloud(PC_PATH)  # Load point cloud
        # if pcd.is_empty():
        #     raise ValueError(f"Failed to load point cloud from {PC_PATH}")
        # self.pc_frame = np.asarray(pcd.points)  # Convert to NumPy array if needed
        # print("Point Cloud loaded successfully.")

        self.depth_frame = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
        self.depth_frame = self.depth_frame.copy()

        self.depth_values = np.loadtxt(DEPTHCSV_PATH, delimiter=",", skiprows=0)
        print(self.depth_values.shape)

    def label_loaded_frames(self, idxs_to_label, text_prompt=TEXT_PROMPT, show_plot=False):
        """Label frames using GSAM2 components."""
        self.seg_frame_lst = []
        # frame = self.video_frame[:, :, ::-1]  # Convert to BGR
        tqdm.write(f"\nProcessing frame...")

        frame, image = load_image(IMG_PATH)

        # sam2_predictor.set_image(image_source)

        # Load the frame into the SAM2 predictor
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
        boxes = boxes * torch.Tensor([w, h, w, h])  # Scale boxes
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Predict masks with SAM2
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Save segmented channels
        seg_frame = np.zeros((frame.shape[0], frame.shape[1], len(labels)))
        for ch_idx in range(len(labels)):
            seg_frame[:, :, ch_idx] = masks[ch_idx]
        

        # Save the segmented frame
        # self.seg_frame_lst.append(seg_frame)

        self.seg_frame_lst = seg_frame

        print(type(self.seg_frame_lst))
        print(type(seg_frame))

        if show_plot:
            # Visualize results
            annotated_frame = sv.MaskAnnotator().annotate(
                scene=frame, 
                detections=sv.Detections(
                    xyxy=input_boxes, 
                    mask=masks.astype(bool), 
                    class_id=np.arange(len(labels)),
                )
            )
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        tqdm.write("Frame processed successfully.")


    # =====================
    # Segmented Frame to PC (local) to PC (global)
    # =====================
    def get_3d_points_from_mask(self, rgb_image, depth_image, segmentation_mask):
        pc_frame = depth_image*10
        fx = 431.0087890625
        fy = 431.0087890625
        ppx = 429.328704833984
        ppy = 242.162155151367

        points_3d = []
        height, width = depth_image.shape

        for v in range(height):
            for u in range(width):
                if segmentation_mask[v, u] > 0:  # If the pixel is part of the region of interest
                    depth = depth_image[v, u]
                    
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
        # self.masked_video_frame_lst = []
        # self.masked_depth_frame_lst = []  # Initialize depth frame list
        # pc_frame = depth_image*10
        fx = 431.0087890625
        fy = 431.0087890625
        ppx = 429.328704833984
        ppy = 242.162155151367

        points_3d = []

        # Get scaling factors
        scale_x = self.depth_values.shape[1] / self.video_frame.shape[1]  # width scaling factor
        scale_y = self.depth_values.shape[0] / self.video_frame.shape[0]  # height scaling factor

        masked_video_frame = np.zeros_like(self.video_frame)
        masked_depth_frame = np.zeros_like(self.depth_frame)  # Initialize depth frame

        seg_frame = self.seg_frame_lst

        min_depth = 2000

        for ch_idx in range(seg_frame.shape[2]):
            mask = seg_frame[:, :, ch_idx] > 0  # Get the mask for class ch_idx
            
            # For each pixel in the color image, map it to the depth image coordinates
            for y_color in range(self.video_frame.shape[0]):
                for x_color in range(self.video_frame.shape[1]):
                    if mask[y_color, x_color]:  # Check if the pixel is part of the segmentation mask
                        # Map the color image coordinates to the depth image coordinates
                        x_depth = int(x_color * scale_x)
                        y_depth = int(y_color * scale_y)
                        
                        # Apply the mask to the RGB frame: only keep values where mask is True
                        masked_video_frame[y_color, x_color] = self.video_frame[y_color, x_color]
                        
                        # Apply the mask to the depth frame: only keep depth values where mask is True
                        masked_depth_frame[y_depth, x_depth] = self.depth_frame[y_depth, x_depth]

                        depth = self.depth_values[y_depth, x_depth]
                    
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
            
            print("HERE")

        self.masked_video_frame = masked_video_frame
        self.masked_depth_frame = masked_depth_frame
        self.masked_pc_frame = points_3d

        print(min(points_3d))

        print(min_depth)

        # self.masked_pc_frame = self.get_3d_points_from_mask(self.video_frame, self.depth_values, self.seg_frame_lst)

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

        # Iterate over each channel (class) and save it
        for ch_idx in range(self.seg_frame_lst.shape[2]):
            # Extract each channel (2D array)
            channel_image = self.seg_frame_lst[:, :, ch_idx]
            
            # Save each channel as a separate PNG
            plt.imsave(f"{save_fpath}/seg_frame_lst_channel_{ch_idx}.png", channel_image, cmap='gray')
    

        # Save masked_video_frame
        if self.masked_video_frame is not None:
            plt.imsave(f"{save_fpath}/masked_video_frame.png", self.masked_video_frame.astype(np.uint8))

        # Save masked_depth_frame
        if self.masked_depth_frame is not None:
            plt.imsave(f"{save_fpath}/masked_depth_frame.png", self.masked_depth_frame.astype(np.uint8))

        if self.masked_pc_frame is not None and len(self.masked_pc_frame) > 0:
            # Convert to Open3D PointCloud
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.masked_pc_frame)

            # Save as PLY file
            o3d.io.write_point_cloud(f"{save_fpath}/masked_pc_frame.ply", pc)

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
    print(sys.argv)
    bag_name = sys.argv[1]
    # bag_name = 'V2DataRedo_realsense0801_lag.bag'

    # calib_fac = sys.argv[2]
    # bag_name = 'realsense_2022-08-01-17-25-56_lag.bag'
    # bag_name = 'realsense_2022-07-12-19-26-18_lag.bag'
    # bag_name = 'human_2022-04-08-14-23-31_lag.bag'
    # bag_name = 'human_2022-04-01-16-30-44_lag.bag'
    # bag_name = 'human_2022-03-06-14-51-06_lag.bag'
    # bag_name = 'field_2021-12-09-16-45-58lag.bag'
    # bag_name = 'V2DataRedo_field_lag.bag'

    t1 = time.time()
    labeler = semantic_labeler(bag_name)
    t2 = time.time()
    print("Loading took: %2.2f seconds"%(t2-t1))
    
    # loop over all video frames
    # idxs_to_label = np.unique(labeler.data_dict['data_array'][:,-1])[2000::20]

    # only loop over all video frames that has a corresponding pc frames 
    idxs_to_label = labeler.pc_video_idxs()
    labeler.label_loaded_frames(idxs_to_label, show_plot=False)
    labeler.apply_segmentation_to_frames()
    
    # remove the ones we don't want to save
    # minimal = True
    # if minimal:
    #     del labeler.data_dict['depth_frame']
    #     del labeler.data_dict['video_frame']
    #     del labeler.data_dict['pc_frame']
    #     del labeler.data_dict['seg_frame']
    #     import gc
    #     gc.collect()
    #     # labeler.intensity_labeling()
    # else:
    #     labeler.intensity_labeling(seg=True)

    labeler.save_frames()