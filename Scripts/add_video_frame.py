import numpy as np
import pickle
import rosbag
import yaml
from tqdm import tqdm
import sensor_msgs
from sensor_msgs.msg import Image
import ros_numpy as rnp
import time
import sys
import struct
from skimage.transform import resize
import joblib
import cv2   # for debug visualization

# ====================
# rnp converts rgb into a float, we need to convert it back to rgb values
def fixrgb(rnp_pc):
    # rnp_pc = 1d array for old dataset, for new dataset it's 480x848 array.
    if len(rnp_pc.shape) == 2:
        rnp_pc = rnp_pc.flatten()
    xyz_lst = np.column_stack((rnp_pc['x'],rnp_pc['y'],rnp_pc['z']))
    float_lst = rnp_pc['rgb']

    # Convert the float32 array to its byte representation
    byte_repr = float_lst.copy().view(np.uint8).reshape(-1, 4)
    rgb_array = byte_repr[:,:3].astype(float)
    
    fixed_pc = np.column_stack((xyz_lst, rgb_array))
    return fixed_pc

# convert a point cloud of shape (n,3) to a depth image chnnel of shape (480,640). 
# Depth is defined as the z distance from the camera to the pixel
def point_to_channelD435(pc_frame, hfov=60, vfov=45):
    hpix,vpix = 640, 480
    # ===============reproject pc to video=========
    curr_depth_frame = np.zeros([vpix,hpix,1])
    ver_ang = np.rad2deg(np.arctan2(pc_frame[:,1], pc_frame[:,2]))
    hor_ang = np.rad2deg(np.arctan2(pc_frame[:,0], pc_frame[:,2]))
    depth   = pc_frame[:,2]

    u,v = (hor_ang/hfov*hpix + hpix/2).astype(int), (ver_ang/vfov*vpix+vpix/2).astype(int)
    curr_depth_frame[v,u,:] = depth[:, np.newaxis]/10

    # ==============Align and remove black margins==========
    # print(min(u),max(u), min(v), max(v))
    curr_depth_frame = curr_depth_frame[2:460, 22:612, :]

    resized_image = resize(curr_depth_frame, (480,618,3), order=0, preserve_range=True)
    pad_width = ((0, 0), (22, 0), (0, 0))
    # Pad the image with zeros on the left (22 pixels)
    curr_depth_frame = np.pad(resized_image, pad_width=pad_width, mode='constant', constant_values=0)
    return curr_depth_frame

def point_to_channelD455(pc_frame):
    depth_fx = 431.0087890625
    depth_fy = 431.0087890625
    depth_ppx = 429.328704833984
    depth_ppy = 242.162155151367
    image_width = 848
    image_height = 480
    # ===============reproject pc to video=========
    # Initialize an empty depth frame
    depth_frame = np.zeros((image_height, image_width, 1))
    pc_frame = pc_frame[pc_frame[:,2]>0.3]

    x = pc_frame[:, 0]
    y = pc_frame[:, 1]
    z = pc_frame[:, 2]
    u = np.round(depth_fx * x / z + depth_ppx).astype(int)
    v = np.round(depth_fy * y / z + depth_ppy).astype(int)

    depth_frame[v, u, :] = z[:, np.newaxis]/10.0

    return depth_frame

def find_scale_shift(Dgt, Dpred):
    """
    Finds the scale and shift coefficients to match the predicted depth image to the ground truth.

    Parameters:
    - Dgt: numpy.ndarray, the ground truth depth image of shape (H, W).
    - Dpred: numpy.ndarray, the predicted relative depth image of shape (H, W).

    Returns:
    - s: float, the scale coefficient.
    - c: float, the shift coefficient.
    """
    mask = np.logical_and(Dgt > 0, Dpred > 0)
    # Flatten the images to use them in a least squares problem
    Dgt_flat = Dgt[mask].flatten()
    Dpred_flat = Dpred[mask].flatten()

    # Create the design matrix A with the predicted depths and a column of ones.
    # Solve the least squares problem Ax = b to find the best fitting s and c
    A = np.vstack([Dpred_flat, np.ones_like(Dpred_flat)]).T
    return np.linalg.lstsq(A, Dgt_flat, rcond=None)[0]

def pc_video_idxs(pc_t, video_t):
    idxs    = []
    for i in range(len(pc_t)):
        curr_t   = pc_t[i]
        curr_idx = np.argmin(abs(video_t-curr_t))
        idxs.append(curr_idx)

    return np.array(idxs, dtype=int)

'''
dataline definition
self.data_line = [msg_t] + self.temp_pos + self.temp_pose + self.temp_pose_var + 
                             self.temp_vel + self.temp_ang_vel + 
                             [self.temp_step] + self.temp_joint + 
                             [self.pc_idx, self.pano_idx, self.video_idx]
'''
class Data_Processor:
    def __init__(self, bag_name):
        self.bag_name = bag_name
        self.load_training_set()
        self.load_bag_file()

    # Load training set
    def load_training_set(self):
        fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/'+'20HZVZS_' + self.bag_name[:-4]
        print("loading "+fpath)

        # infile = open(fpath,'rb')
        # self.data_dict = pickle.load(infile, encoding='latin1')
        self.data_dict = joblib.load(fpath)
        data_arr_t_st, data_arr_t_ed = self.data_dict['data_array'][0,0], self.data_dict['data_array'][-1,0]

        print("\n\nDone loading, panorama length:")
        print(len(self.data_dict['pano_t']))
        print(self.data_dict.keys())
        print('start', self.data_dict['pano_t'][0], 'end', self.data_dict['pano_t'][-1])

        self.pano_duration = self.data_dict['pano_t'][-1] - self.data_dict['pano_t'][0]
        print('duration', self.pano_duration)
        print(self.data_dict['pano_frame'][0].shape)

        # =======Test=========
        # print('='*20+'Test Area'+'='*20)
        # print(self.data_dict['data_array'].shape)
        # print('t0', self.data_dict['data_array'][0,:])
        # ====================

    def load_bag_file(self):
        # Load bag file
        fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Bags/' + self.bag_name
        print("\n\nloading bag file" + fpath)
        self.bag = rosbag.Bag(fpath)
        info_dict = yaml.safe_load(self.bag._get_yaml_info())
        print(info_dict.keys())

    '''
    This function is used to fill in pano_t array and pano_idx in data_array, so that DINO labeler can 
    renegerate the panorama at the correct time. 
    This is only needed when the panorama is not fully generated, but the point cloud is correctly saved.
    '''
    def fix_pano_idx(self):
        # Check the index to replace
        pano_idx = self.data_dict['data_array'][:,-2]
        pano_t = self.data_dict['pano_t']
        replace_idx_start = np.where(pano_idx==pano_idx[-1])[0][1:]
        if len(replace_idx_start)<100:
            raise ValueError('ERROR: pano_idx is already fully generated, no need to fix, please double check')
        else:
            for i in range(len(replace_idx_start)):
                curr_da_idx = replace_idx_start[i]                         # current data array index
                self.data_dict['data_array'][curr_da_idx,-2] = len(pano_t) # replace with correct panorama index
                pano_t.append(self.data_dict['data_array'][curr_da_idx,0])
        
        print('Done fixing panorama index')


    def insert_video_and_pc(self):
        print('Scanning for video and point cloud frames')
        # Populate with pc and video frames
        video_t, video_frame, pc_t, pc_frame = [], [], [], []
        with tqdm(total=np.floor(self.pano_duration*32)) as pbar:
            for topic, msg, t in self.bag.read_messages(): # topics=['/step_counter','/rosout']
                if (topic in ["/d400/color/image_raw","/camera/color/image_raw"]):
                    # print('[TYPE] video_frame')
                    # pbar.write('[TYPE] video_frame')
                    
                    # Convert to cv frame
                    msg.__class__ = sensor_msgs.msg._Image.Image
                    img = rnp.numpify(msg)
                    # img = np.transpose(img,(2,0,1))
                    
                    # Gather info from header
                    msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9

                    # Save to respective array
                    video_t.append(msg_t)
                    video_frame.append(img)
                    # video_idx = len(video_t)-1
                    pbar.update(1)

                if (topic in ['/save_pc']):
                    # pbar.write('got pc')
                    msg.__class__ = sensor_msgs.msg.PointCloud2
                    pc_data = fixrgb(rnp.numpify(msg))
                    pc_t.append(t.to_sec())
                    pc_frame.append(pc_data)

                # if (topic in ['/rosout']):
                #     print(msg.msg)

        print('Done extracting video frames, begin inserting')

        # insert frame into dataset
            # insert data in dict
        self.data_dict['video_t'] = video_t
        self.data_dict['video_frame'] = video_frame
        self.data_dict['pc_frame'] = pc_frame
        self.data_dict['pc_t'] = pc_t

            # insert idx in data array
        data_array = self.data_dict['data_array']
        for i in tqdm(range(data_array.shape[0])):
            curr_t = data_array[i,0]
            curr_cam_idx = np.sum(video_t<curr_t)-1
            curr_pc_idx = np.sum(pc_t<curr_t)-1

            data_array[i,-1] = curr_cam_idx
            data_array[i,-3] = curr_pc_idx

        self.data_dict['data_array'] = data_array
        print('Done inserting point cloud and video frames')
        
    def pc_to_depth(self):
        print('Converting point cloud to depth frames')
        depth_frame = []    # this frame is aligned with pc_frame, share the same pc_t
        pc_frame = self.data_dict['pc_frame']
        for i in tqdm(range(len(pc_frame))):
            depth_img = point_to_channelD455(pc_frame[i])
            depth_frame.append(depth_img)
        self.data_dict['depth_frame'] = depth_frame

    def save_to_disk(self):
        # save as new dataset
        print('Saving to disk...')
        t_start = time.time()

        save_fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/V20HZVZS_' + self.bag_name[:-4]
        # outfile = open(save_fpath,'wb')
        # pickle.dump(self.data_dict, outfile)
        # outfile.close()
        joblib.dump(self.data_dict, save_fpath, compress=('lz4', 1))

        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))

    # replace all depth frames with learned depth
    def redo_learned_depth_zoe(self):
        import torch
        Zoe_path = '/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/ZoeDepth'
        sys.path.insert(1, Zoe_path)
        from zoedepth.utils.misc import get_image_from_url, colorize
        from PIL import Image

        zoe = torch.hub.load("/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/ZoeDepth/", "ZoeD_NK", source="local", pretrained=True)
        zoe = zoe.to('cuda')
    
        # prepare data
        depth_t = np.array(self.data_dict['pc_t'])
        video_t = np.array(self.data_dict['video_t'])
        idxs    = pc_video_idxs(depth_t, video_t)       # corresponding video index for each point cloud frame
        rgb_frames = np.array(self.data_dict['video_frame'], dtype=np.float32)[idxs]/255.0
        old_depth_frames = np.array(self.data_dict['depth_frame'])

        print('Redoing with learned ZoeDepth')
        for i in tqdm(range(len(self.data_dict['depth_frame']))):
            # Prepare data
            image = rgb_frames[i]
            old_depth = np.round(old_depth_frames[i,:,:,0]*255)
            h, w = image.shape[:2]
            
            image = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to('cuda')
            
            # Generate depth
            # input: [1,3,H,W]
            t1 = time.time()
            depth = zoe.infer(image)
            t2 = time.time()
            print('Inference time:', t2-t1)

            depth = depth.detach().cpu().numpy()[0,0]
            
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.cpu().numpy().astype(np.uint8)

            # # compare with old depth
            new_depth = depth.copy()
            # new_depth[old_depth<1] = 0
            s,c = find_scale_shift(old_depth, new_depth)
            new_depth_grounded = depth*s+c

            # # clamp values
            new_depth_grounded[new_depth_grounded<0] = 0
            new_depth_grounded[new_depth_grounded>150] = 0 # remove inaccurate values

                        # ====+Second Matching+=======
            s,c = find_scale_shift(old_depth, new_depth_grounded)
            print("Second scaling:",s," ",c)
            new_depth_grounded = new_depth_grounded*s+c

            # clamp values
            new_depth_grounded = np.round(new_depth_grounded)
            new_depth_grounded[new_depth_grounded<0] = 0
            new_depth_grounded[new_depth_grounded>255] = 255

            # remove diverging pixels
            diff = np.abs(new_depth_grounded-old_depth)
            mask = np.logical_and(diff>50, old_depth>0)
            new_depth_grounded[mask] = 0
            cv2.imshow('new', new_depth_grounded/255)
            cv2.waitKey(1)
            self.data_dict['depth_frame'][i] = new_depth_grounded[:,:,np.newaxis]/255.0

    # replace all depth frames with learned depth
    def redo_learned_depth(self):
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import Compose
        DA_path = '/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/Depth-Anything'
        sys.path.insert(1, DA_path)
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        # load model
        encoder_name = 'vitl'
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder_name)).to('cuda').eval()
        
        # Preprocessing pipeline
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        depth_t = np.array(self.data_dict['pc_t'])
        video_t = np.array(self.data_dict['video_t'])
        idxs    = pc_video_idxs(depth_t, video_t)       # corresponding video index for each point cloud frame
        rgb_frames = np.array(self.data_dict['video_frame'])[idxs]/255.0
        old_depth_frames = np.array(self.data_dict['depth_frame'])

        print('Redoing with learned depth')
        for i in tqdm(range(len(self.data_dict['depth_frame']))):
            # Prepare data
            image = rgb_frames[i]
            old_depth = np.round(old_depth_frames[i,:,:,0]*255)
            h, w = image.shape[:2]
            
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to('cuda')
            
            # Generate depth
            with torch.no_grad():
                depth = depth_anything(image)
            
            depth = F.interpolate(depth[None], (h, w), mode='nearest')[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth.cpu().numpy()

            # compare with old depth
            new_depth = depth.copy()
            # new_depth[old_depth<1] = 0
            s,c = find_scale_shift(old_depth, new_depth)
            new_depth_grounded = depth*s+c

            # clamp values
            new_depth_grounded = np.round(new_depth_grounded)
            new_depth_grounded[new_depth_grounded<0] = 0
            new_depth_grounded[new_depth_grounded>255] = 255
            # cv2.imshow('new', new_depth_grounded/255)
            # cv2.waitKey(1)
            self.data_dict['depth_frame'][i] = new_depth_grounded[:,:,np.newaxis]/255.0

    def remove_edges(self):
        print("Removing edges")
        depth_frame = np.array(self.data_dict['depth_frame']).repeat(3, axis=3)
        for i in tqdm(range(len(depth_frame))):
            a = depth_frame[i]

            # Canny edge detection
            edges = cv2.Canny(np.round(a*255).astype(np.uint8), 25, 32)

            # expand edges
            kernel = np.ones((10,10))
            eroded_edges = cv2.erode(edges-1,kernel)+1
            eroded_edges = eroded_edges[:,:,np.newaxis].repeat(3, axis=2)

            # remove edges
            processed_frame = a.copy()
            processed_frame[eroded_edges>0] = 0

            # cv2.imshow('processed', processed_frame)
            # cv2.waitKey(1)
            # put back
            self.data_dict['depth_frame'][i] = processed_frame[:,:,[0]]

if __name__ == "__main__":

    print(sys.argv)
    # load training set
    # bag_name = 'realsense_2022-08-01-17-25-56_lag.bag'
    # bag_name = 'realsense_2022-07-12-19-26-18_lag.bag'
    # bag_name = 'human_2022-04-08-14-23-31_lag.bag'
    # bag_name = 'human_2022-04-01-16-30-44_lag.bag'
    # bag_name = 'human_2022-03-06-14-51-06_lag.bag'
    # bag_name = 'field_2021-12-09-16-45-58lag.bag'
    # bag_name = 'V2DataNew_231228lawold_needpanofix_lag.bag'
    # bag_name = 'V2DataRedo_realsense0801_lag.bag'
    bag_name = sys.argv[1]


    # bag_name = 'V2DataRedo_field_lag.bag'

    dp = Data_Processor(bag_name)
    if '--reindex' in sys.argv:
        print('Reindexing...')
        dp.fix_pano_idx()               # only call this when panorama is not fully generated, but pc is saved correctly
    
    dp.insert_video_and_pc()
    dp.pc_to_depth()
    # dp.redo_learned_depth_zoe()
    dp.remove_edges()
    dp.save_to_disk()