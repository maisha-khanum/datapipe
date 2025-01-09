# Need to use the GSAM env for this script

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import torch, time, pickle, joblib

from typing import Tuple
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from segment_anything import build_sam, SamPredictor 
from skimage.transform import resize
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed

import sys
GSAM_path = '/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/Grounded-Segment-Anything'
sys.path.insert(1, GSAM_path)
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict

# ==========================General Helper Functions==========================
SURROUND_U_STEP = 1.    #resolution
SURROUND_V_STEP = 1.
SURROUND_U_MIN, SURROUND_U_MAX = np.array([0,    360])/SURROUND_U_STEP  # horizontal of cylindrial projection
SURROUND_V_MIN, SURROUND_V_MAX = np.array([-90,   90])/SURROUND_V_STEP  # vertical   of cylindrial projection
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GDINO helper
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cuda')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# SAM helper
def show_mask(mask, image, random_color=True, color=None):
    if color is None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


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
    return np_image, image_transformed

def generate_pano(pc_frame_glob, curr_pos,r,sample_ratio,view_dist=10.0,filter=False):    

    total_pt_array = np.vstack([pc_frame[0] for pc_frame in pc_frame_glob])
    remain_idx = np.random.permutation(np.arange(total_pt_array.shape[0]))[:int(total_pt_array.shape[0]*sample_ratio)]
    total_pt_array = total_pt_array[remain_idx,:]

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
    surround_img = np.zeros((height, width,10)) # rgbd c1-c6
    surround_img[:,:,3] = 255 # d=255 if unknown

    @nb.njit # OPTIMIZED
    def proj_and_clip(tmp, u,v,d, total_pt_array): # OPTIMIZED
        for i in range(len(u)):
            if tmp[v[i],u[i],3] > d[i]:
                _, _, _, r,g,b, c1,c2,c3,c4,c5,c6 = total_pt_array[i,:]
                tmp[v[i],u[i],:] = np.array([r,g,b, d[i], c1,c2,c3,c4,c5,c6])

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

# ==========================Labeler Class==========================
class semantic_labeler:
    def __init__(self, bag_name):
        self.device='cuda'
        self.bag_name = bag_name

        self.load_GDINO_SAM()
        self.load_data_source()
    
    # ============
    # Segmentation
    # ============

    def load_GDINO_SAM(self):
        # ==============Load G DINO================

        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        print('Loading G DINO')
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


        # ==============Load SAM================
        print('\n\nLoading SAM')
        sam_checkpoint = GSAM_path+'/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        # return groundingdino_model, sam_predictor

    def load_data_source(self):
        fpath = '/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/'+'V20HZVZS_' + self.bag_name[:-4]
        print("\nLoading "+fpath)
        # infile = open(fpath,'rb')
        # self.data_dict = pickle.load(infile, encoding='latin1')
        # infile.close()
        self.data_dict = joblib.load(fpath)


    def detect_by_text(self, image, image_source, TEXT_PROMPT, BOX_TRESHOLD=0.3, TEXT_TRESHOLD=0.25):
        
        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return annotated_frame, boxes, logits, phrases
    
    def segment_by_detection(self, image_source, boxes, reuse_enc=False):
        # set image
        if not reuse_enc:
            self.sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        
        image_mask = torch.sum(masks,axis=0,dtype=bool)
        # image_mask = masks[0][0]
        return image_mask

    def label_loaded_frames(self, idxs_to_label, show_plot=False):
        self.seg_frame_lst = []
        for idx in tqdm(idxs_to_label):
            frame = self.data_dict['video_frame'][int(idx)]
            image_source, image = load_frame(frame)
            new_frame = True
            seg_frame = np.zeros((image_source.shape[0],image_source.shape[1],len(prompts)))
            tqdm.write(f'\n')
            # ==============Run G DINO to detect================
            masked_source = image_source.copy()
            for idx, (TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, color) in enumerate(prompts):
                t2 = time.time()
                annotated_frame, boxes, logits, phrases = labeler.detect_by_text(image, image_source,
                                                                                TEXT_PROMPT, 
                                                                                BOX_TRESHOLD=BOX_TRESHOLD, 
                                                                                TEXT_TRESHOLD=TEXT_TRESHOLD)
                print("Input phrases:", TEXT_PROMPT, "Phrases detected:", phrases)
                t23 = time.time()
                print(logits)
                # ==============Segment with detection================
                tqdm.write(f'{boxes.shape[0]} {TEXT_PROMPT}')
                if boxes.shape[0] > 0.1:
                    for box_idx in range(boxes.shape[0]):
                        image_mask = labeler.segment_by_detection(image_source, boxes[box_idx], reuse_enc=False if new_frame else True)
                        seg_frame[:,:,idx] = np.maximum(seg_frame[:,:,idx], (image_mask[0]*logits[box_idx]).cpu().numpy()) # element wise max
                        new_frame = False
                        masked_source = show_mask(image_mask, masked_source, color=np.array([color[0],color[1],color[2],color[3]*logits[box_idx]]))
                else:
                    tqdm.write(f"----No {TEXT_PROMPT} found----")

                t3 = time.time()
                tqdm.write(f"Prediction took: {t3-t2:.2f} s, seg: {t3-t23:.4f} s, det: {t23-t2:.2f} s")

            # save segmented channel frame
            self.seg_frame_lst.append(seg_frame)

            if show_plot:
                plt.clf()
                plt.imshow(masked_source)
                plt.axis('off')
                plt.pause(0.3)

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

    def depth_to_pc(self, depth_frame, seg_frame, video_frame, hfov=60, vfov=45):
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
                        c1,c2,c3,c4,c5,c6 = seg_frame[v,u,:]

                        # save to dictionary
                        # pt_dict[(v,u)] = [x,y,z, r,g,b, c1,c2,c3,c4,c5,c6]
                        # keys.append((v,u))
                        values.append([x,y,z, r,g,b, c1,c2,c3,c4,c5,c6])
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
        sample_ratio = 0.14
        pano_t = self.data_dict['pano_t']
        view_dist = 10.0
        window_sz = 32

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
        self.redo_pano_frame = Parallel(n_jobs=10,backend="threading")(delayed(make_pano_single_pass)(pano_t[i],sample_ratio) for i in range(n_keypose)) # n_keypose

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

    def save_training_set(self):
        self.data_dict['prompts'] = prompts
        
        # remove the ones we don't want to save
        minimal = True
        if minimal:
            self.data_dict['depth_frame'] = []
            self.data_dict['seg_frame'] = []  
            self.data_dict['video_frame'] = []
            self.data_dict['pc_frame'] = []
        # save as new dataset
        print('Saving to disk...')
        t_start = time.time()

        save_fpath = '../Training_sets/S20HZVZS_' + self.bag_name[:-4]
        # outfile = open(save_fpath,'wb')
        # pickle.dump(self.data_dict, outfile)
        # outfile.close()
        joblib.dump(self.data_dict, save_fpath, compress=('gzip', 0))

        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))

if __name__ == '__main__':
    # Prompt, how confident is the box, how closely should it match text, rgba
    prompts = [["ground, sidewalk",0.2,0.28,[0.1, 0.1, 0.1, 0.7]], # dark
               ["stairs",0.4,0.4,[0.7, 0.7, 0.7, 0.7]], # gray ish
        #    ["ramp",0.3,0.25,[0.5, 0.5, 0.5, 0.7]],   # dark ish
               ["door, wood door, steel door, glass door, elevator door",0.55,0.5,[0.0, 0.7, 0.0, 0.7]],   # green
               ["wall, pillar",0.47,0.3,[0.7, 0.0, 0.0, 0.7]],   # red
               ["bin, chair, bench, desk, plants, curb, bushes, pole, tree",0.55,0.5,[0.0, 0.0, 0.7, 0.7]],
               ["people, person, pedestrian", 0.5, 0.5, [0.0, 0.0, 0.7, 0.7]],
               ["grass, field, sand, hill, earth, dirt", 0.5, 0.5, [0.5,0.5,0.5,0.7]]] # blue

    # ==============Load model and image to label================
    # Load data
    print(sys.argv)
    bag_name = sys.argv[1]
    # calib_fac = sys.argv[2]
    # bag_name = 'realsense_2022-08-01-17-25-56_lag.bag'
    # bag_name = 'realsense_2022-07-12-19-26-18_lag.bag'
    # bag_name = 'human_2022-04-08-14-23-31_lag.bag'
    # bag_name = 'human_2022-04-01-16-30-44_lag.bag'
    # bag_name = 'human_2022-03-06-14-51-06_lag.bag'
    # bag_name = 'field_2021-12-09-16-45-58lag.bag'

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
    labeler.save_training_set()