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

from typing import Any, Tuple
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from segment_anything import build_sam, SamPredictor 
from skimage.transform import resize
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed

import mmcv, urllib
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
import dinov2.eval.segmentation_m2f.models.segmentors

import GroundingDINO.groundingdino.datasets.transforms as T

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

TEXT_PROMPT = "stairs."
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_frame(np_image: np.array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([480], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # # image_source = Image.fromarray(np.transpose(np_image,(1,2,0))).convert("RGB")
    image_source = Image.fromarray(np_image).convert("RGB")
    # image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    print(image_transformed.shape)
    return np_image, image_transformed

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

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

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

    def load_data_source(self):
        """Load data source based on the provided bag name."""
        fpath = '/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/' + 'V20HZVZS_' + self.bag_name[:-4]
        print(f"\nLoading {fpath}")
        self.data_dict = joblib.load(fpath)
        print("Data source loaded successfully.")

    def label_loaded_frames(self, idxs_to_label, text_prompt=TEXT_PROMPT, show_plot=False):
        """Label frames using GSAM2 components."""
        self.seg_frame_lst = []
        for idx in tqdm(idxs_to_label):
            frame = self.data_dict['video_frame'][int(idx)][:, :, ::-1]  # Convert to BGR
            tqdm.write(f"\nProcessing frame {idx}...")

            # Load the frame into the SAM2 predictor
            self.sam2_predictor.set_image(frame)

            # Predict bounding boxes and masks using Grounding DINO
            h, w, _ = frame.shape
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=frame,
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
            self.seg_frame_lst.append(seg_frame)

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

            tqdm.write(f"Frame {idx} processed successfully.")

    def dump_results_to_json(self, output_dir=OUTPUT_DIR):
        """Dump segmentation results to JSON format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for idx, seg_frame in enumerate(self.seg_frame_lst):
            masks = [single_mask_to_rle(seg_frame[:, :, ch]) for ch in range(seg_frame.shape[2])]
            results.append({"frame_index": idx, "masks": masks})

        json_path = output_dir / "segmentation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {json_path}.")

    # =====================
    # Segmented Frame to PC (local) to PC (global)
    # =====================

    def post_process_seg_frame(self, video_idxs_to_label):
        print("Converting segmented frames to global point clouds")
        self.pc_frame_glob = [None]*len(self.seg_frame_lst)

        # def process_single_frame(i, video_idxs_to_label):
        #     print('processing:',i)
        #     # Only keep the area with depth info
        #     video_t = self.data_dict['video_t'][video_idxs_to_label[i]]
        #     depth_idx = np.argmin(abs(np.array(self.data_dict['pc_t'])-video_t))
        #     video_frame = self.data_dict['video_frame'][video_idxs_to_label[i]]
        #     depth_frame = self.data_dict['depth_frame'][depth_idx]

        #     seg_frame = self.seg_frame_lst[i]
        #     seg_frame = seg_frame*np.sign(depth_frame[:,:,[0]])

        #     # Reproject to point cloud in global frame
        #     t0 = time.time()
        #     pc_local_k, pc_local_v  = self.depth_to_pc(depth_frame, seg_frame, video_frame) # PC in local frame
        #     t1 = time.time()
        #     pc_global = self.local_pc_to_global(pc_local_k, pc_local_v, self.data_dict['pc_t'][depth_idx]) # PC in global frame
        #     t2 = time.time()
        #     # print("Taken total",t2-t1, t1-t0)

        #     return pc_global
        
        # self.pc_frame_glob = Parallel(n_jobs=5)(delayed(process_single_frame)(i, video_idxs_to_label) for i in range(len(self.seg_frame_lst)))

        for i in tqdm(range(len(self.seg_frame_lst))):
            # Only keep the area with depth info
            video_t = self.data_dict['video_t'][video_idxs_to_label[i]]
            depth_idx = np.argmin(abs(np.array(self.data_dict['pc_t'])-video_t))
            video_frame = self.data_dict['video_frame'][video_idxs_to_label[i]]
            depth_frame = self.data_dict['depth_frame'][depth_idx]

            seg_frame = self.seg_frame_lst[i]
            seg_frame = seg_frame*np.sign(depth_frame[:,:,[0]])

            # Reproject to point cloud in global frame
            t0 = time.time()
            pc_local_v  = self.depth_to_pc(depth_frame, seg_frame, video_frame) # PC in local frame
            t1 = time.time()
            pc_global_v = self.local_pc_to_global(pc_local_v, self.data_dict['pc_t'][depth_idx]) # PC in global frame
            t2 = time.time()
            del pc_local_v
            # print("Taken total",t2-t1, t1-t0)
            self.pc_frame_glob[i] = [pc_global_v]     # Add to list

        print("Done")

        print("Begin regenerating panorama")
        self.redo_pano()

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

    def local_pc_to_global(self, pc_local_v, curr_t):
        if len(pc_local_v) == 0:
            return np.array([-100,-100,-100]) # return a dummy point far away that will have no effect
        t1 = time.time()
        curr_idx = np.argmin(abs(self.data_dict['data_array'][:,0]-curr_t))
        curr_pos, curr_quat = self.data_dict['data_array'][curr_idx, 1:4], self.data_dict['data_array'][curr_idx, 4:8]
        rot = R.from_quat(curr_quat).as_matrix()
        
        # convert all points to global frame
        keys, values = [], []
        pcl_values = np.array(pc_local_v)
        pt = np.array([pcl_values[:,2], -pcl_values[:,0], -pcl_values[:,1]]).T # from realsense definition to our definition
        global_pt = (rot.dot(pt.T)).T + curr_pos
        pcl_values[:,:3] = global_pt
        t2 = time.time()
        # print("Taken",t2-t1)
        return pcl_values


        # for i in pc_local.keys():
        #     x,y,z, r,g,b, c1,c2,c3,c4,c5,c6 = pc_local[i]
            
        #     pt = np.array([[z, -x, -y]]) # from realsense definition to our definition

        #     # Convert to global frame
        #     global_pt = rot.dot(pt.T)
        #     global_pt = global_pt.T + curr_pos

        #     keys.append(i)
        #     values.append([global_pt[0,0],global_pt[0,1],global_pt[0,2], 
        #                 r,g,b, c1,c2,c3,c4,c5,c6])

        # return dict(zip(keys, values))

    # =====================
    # Panorama
    # =====================

    def redo_pano(self):
        sample_ratio = 0.5    # 0.2, 0.3, 0.5 for high accuracy
        pano_t = self.data_dict['pano_t']
        view_dist = 8.0
        window_sz = 9  # 32

        def make_pano_single_pass(curr_pano_t, sample_ratio):
            print('[INFO] Generating Panorama for time {}/{}'.format(sum(np.array(pano_t)<curr_pano_t) , n_keypose))
            # ===Get pose to generate panorama===
            curr_pos,curr_quat = self.find_nearest_pos(curr_pano_t)
            key_pose = np.append(curr_pos,curr_quat)
            print('[INFO] Keypose location:',key_pose[:3])

            # ===Prepare necessary information===
            t1 = time.time()
            # r = R.from_quat(key_pose[3:]).as_matrix() # Old way, follow camera

            # New way, gopro like horizon locked
            r = R.from_quat(key_pose[3:]).as_euler('ZYX')
            r = R.from_euler('ZYX',[r[0],0,0]).as_matrix()

            idx = sum(np.array(self.data_dict['pc_t'])<curr_pano_t)                                 # most up to date point cloud index
            window_l, window_r = max(idx-window_sz+1,0), idx+1     # Only input window data
            print(window_l,window_r)
            # This step requires you to process all of the point cloud data. Otherwise it would be mismatch and out of bound
            panorama = generate_pano(self.pc_frame_glob[window_l:window_r], key_pose[:3],r,sample_ratio,view_dist=view_dist,filter=True)
            # if idx > 0:
            #     plt.clf()
            #     plt.imshow(panorama[:,:,:3]/255)
            #     plt.pause(0.3)
            t2 = time.time()
            print("Taken",t2-t1)
            return panorama
        
        # Generate for each key position in the path
        n_keypose = len(pano_t)
        self.redo_pano_frame = [None]*n_keypose
        self.redo_pano_frame = Parallel(n_jobs=3,backend="threading")(delayed(make_pano_single_pass)(pano_t[i],sample_ratio) for i in range(n_keypose)) # n_keypose

        # for i in range(n_keypose):
        # for i in range(350,370):
            # self.redo_pano_frame[i] = make_pano_single_pass(pano_t[i],sample_ratio)

        self.data_dict['seg_frame'] = self.seg_frame_lst
        self.data_dict['pano_frame'] = np.array(self.redo_pano_frame, dtype=np.float32)

    # =====================
    # General Helper Functions
    # =====================

    def find_nearest_pos(self,t):
        idx = np.argmin(abs(self.data_dict['data_array'][:,0] - t))
        return self.data_dict['data_array'][idx,1:4].copy(), self.data_dict['data_array'][idx,4:8].copy()

    def pc_video_idxs(self):
        pc_t    = np.array(self.data_dict['pc_t'])
        video_t = np.array(self.data_dict['video_t'])
        idxs    = []

        for i in range(len(pc_t)):
            curr_t   = pc_t[i]
            curr_idx = np.argmin(abs(video_t-curr_t))
            idxs.append(curr_idx)

        return np.array(idxs)

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

        save_fpath = '/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_' + self.bag_name[:-4]
        # outfile = open(save_fpath,'wb')
        # pickle.dump(self.data_dict, outfile)
        # outfile.close()
        joblib.dump(self.data_dict, save_fpath, compress=('lz4', 1))

        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))

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
    labeler.post_process_seg_frame(idxs_to_label)
    # remove the ones we don't want to save
    minimal = True
    if minimal:
        # del labeler.data_dict['depth_frame']
        del labeler.data_dict['video_frame']
        del labeler.data_dict['pc_frame']
        del labeler.data_dict['seg_frame']
        import gc
        gc.collect()
        # labeler.intensity_labeling()
    else:
        labeler.intensity_labeling(seg=True)

    labeler.save_training_set()