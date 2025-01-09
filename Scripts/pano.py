import time
from tqdm import tqdm
from numba import jit
import numpy as np
import ros_numpy as rnp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Multi process
from joblib import Parallel, delayed
import multiprocessing

SURROUND_U_STEP = 1.    #resolution
SURROUND_V_STEP = 1.
SURROUND_U_MIN, SURROUND_U_MAX = np.array([0,    360])/SURROUND_U_STEP  # horizontal of cylindrial projection
SURROUND_V_MIN, SURROUND_V_MAX = np.array([-90,   90])/SURROUND_V_STEP  # vertical   of cylindrial projection

# ========================
# Helper functions
# ========================

def lidar_to_surround_coords(x, y, z, dxy):
    u =   np.arctan2(x, y)/np.pi*180 /SURROUND_U_STEP
    v = - np.arctan2(z, dxy)/np.pi*180 /SURROUND_V_STEP
    u = (u + 90 + 360)%360

    u = np.floor(u)
    v = np.floor(v)
    u = (u - SURROUND_U_MIN).astype(np.uint16)
    v = (v - SURROUND_V_MIN).astype(np.uint16)
    return u,v

@jit(nopython=True) # OPTIMIZED
def normalize_to_255(a):
    return (a/10.0*255.0).astype(np.uint8)

def generate_pano(pc_frame_glob, curr_pos,r,view_dist=10.0,filter=False):        
    total_pt_array = np.vstack(pc_frame_glob)

    # r = R.from_quat(curr_quat).as_matrix()
    total_pt_array[:,:3] = total_pt_array[:,:3]-curr_pos
    total_pt_array[:,:3] = total_pt_array[:,:3].dot(r) # rotate to user heading

    x = total_pt_array[:,0]
    y = total_pt_array[:,1]
    z = total_pt_array[:,2]
    # x,y,z = total_pt_array.T

    dxy = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin, this is projected dist onto xy plane
    dist = np.sqrt(dxy**2 + z**2) # xyz distance

    ##### Filter d based on distance #####
    if filter:
        before = len(x)
        x = x[dist<=view_dist]
        y = y[dist<=view_dist]
        z = z[dist<=view_dist]
        dxy = dxy[dist<=view_dist]
        dist = dist[dist<=view_dist]
        after = len(x)
        print('before: ',before,' after: ',after , 'ratio: ', after/before)

    u,v = lidar_to_surround_coords(x,y,z,dxy)

    # # A little trick to make sure we don't see through walls
    # rank_idx = np.argsort(d)[::-1] # find index in descending order
    # # sort all 3 in descending order by d
    # u = u[rank_idx]
    # v = v[rank_idx]
    # d = d[rank_idx]

    width  = int(SURROUND_U_MAX - SURROUND_U_MIN)
    height = int(SURROUND_V_MAX - SURROUND_V_MIN)
    surround_img = np.zeros((height, width), dtype=np.uint8)+255

    @jit(nopython=True) # OPTIMIZED
    def proj_and_clip(tmp, u, v, d): # OPTIMIZED
        for i in range(len(u)):
            if tmp[v[i],u[i]] > d[i]:
                tmp[v[i],u[i]] = d[i]
            # tmp[v[i],u[i]] = min(tmp[v[i],u[i]],d[i])   # Or we can do this to prevent from looking through the wall, slow though

    proj_and_clip(surround_img, u, v, normalize_to_255(np.clip(dist, 0.0, 10.0)))

    return surround_img





# -------------------------------------------------------------------------------------------
class training_set_generator:
    # ...

    def extract_data(self, topic, msg, t, calib_fac):
        # ....

        elif (topic == '/save_pc'):
            # pbar.write('[TYPE] point cloud message')
            msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
            print("PC: bag_time",t,"msg_time",msg_t,"diff",t.to_sec()-msg_t)

            pt_array = rnp.point_cloud2.pointcloud2_to_xyz_array(msg,remove_nans=True) # OPTIMIZED
            pt_array[:,[0,1,2]] = pt_array[:,[0,1,2]] * calib_fac                      # NEW: calibrate the point cloud from source
            d = pt_array[:,0]**2 + pt_array[:,1]**2 + pt_array[:,2]**2
            pt_array = pt_array[d<self.view_dist**2]                                   # Remove points that are too far

            # Save to respective array
            self.pc_t.append(msg_t)
            self.pc_frame.append(pt_array)
            self.pc_idx = len(self.pc_t)-1
            new_entry = True
        
        # ....

    # For each time stamp entry in key pose list, find the corresponding pose in raw data
    def find_nearest_pos(self,t):
        idx = np.argmin(abs(self.data_array[:,0] - t))
        return self.data_array[idx,1:4].copy(), self.data_array[idx,4:8].copy()

    # Transform point array from realsense frame to global frame
    def pc2globalpc(self,pt_array,curr_pos,curr_quat):        
        pt_array[:,[0,1,2]] = pt_array[:,[2,0,1]]
        pt_array[:,1] = -pt_array[:,1]
        pt_array[:,2] = -pt_array[:,2]

        # Convert to global frame
        r = R.from_quat(curr_quat).as_matrix()
        global_xyz = r.dot(pt_array[:,:3].T)
        global_xyz = global_xyz.T + curr_pos

        pt_array[:,:3] = global_xyz
        return pt_array

    # Redo the panorama generation with calibration parameters. It will replace the current panorama with
    # new panorama for the model training data.
    def redo_panorama(self,calib_fac,sample_ratio=0.07):
        # Input check
        if len(self.data_array)<5:
            print("[WARNING] You must call generate_set() before redo_panorama()")
            return
        
        # Convert every pc into global coordinate
        with tqdm(total=len(self.pc_frame)) as pbar:
            for idx, pc in enumerate(self.pc_frame):
                # print('[INFO] Processing point cloud {}/{}'.format(idx+1,len(self.pc_frame)))
                curr_pos,curr_quat = self.find_nearest_pos(self.pc_t[idx])

                # downsample
                n_pts = int(pc.shape[0]*sample_ratio)
                sample_idx = np.random.permutation(pc.shape[0])[:n_pts]
                pc_post = pc[sample_idx,:] #*calib_scale   # NEW: since we already mulitplied it in the raw pc

                pc_global = self.pc2globalpc(pc_post, curr_pos,curr_quat)
                self.pc_frame_glob.append(pc_global)

                pbar.update(1)

        # Collect all panorama in a readable format
        # total_pc_array = np.concatenate(self.pc_frame_glob,axis=0)

        # Generate for each key position in the path
        n_keypose = len(self.pano_t)
        self.redo_pano_frame = [None]*n_keypose

        def make_pano_single_pass(pano_t):
            print('[INFO] Generating Panorama for time {}/{}'.format(sum(np.array(self.pano_t)<pano_t),n_keypose))
            # ===Get pose we use to generate panorama===
            curr_pos,curr_quat = self.find_nearest_pos(pano_t)
            key_pose = np.append(curr_pos,curr_quat)
            print('[INFO] Keypose location:',key_pose[:3])

            # ===Prepare necessary information===
            t1 = time.time()
            # r = R.from_quat(key_pose[3:]).as_matrix() # Old way, follow camera

            # New way, gopro like horizon locked
            r = R.from_quat(key_pose[3:]).as_euler('ZYX')
            r = R.from_euler('ZYX',[r[0],0,0]).as_matrix()

            idx = sum(np.array(self.pc_t)<pano_t)                       # most up to date point cloud index
            window_l, window_r = max(idx-self.window_sz+1,0), idx+1     # Only input window data
            print(window_l,window_r)
            panorama = generate_pano(self.pc_frame_glob[window_l:window_r], key_pose[:3],r,view_dist=self.view_dist,filter=True)
            t2 = time.time()
            print("Taken",t2-t1)
            return panorama

        print(self.par)
        if self.par:
            result = Parallel(n_jobs=7,backend="threading")(delayed(make_pano_single_pass)(self.pano_t[i]) for i in range(n_keypose))
            self.redo_pano_frame = result
        else:
            # Debug only
            for idx in range(n_keypose):
                panorama = make_pano_single_pass(self.pano_t[idx])
                self.redo_pano_frame[idx] = panorama

                plt.clf()
                plt.subplot(121)
                plt.imshow(panorama,cmap='gray',vmin=0,vmax=255)
                plt.subplot(122)
                plt.imshow(self.pano_frame[idx],cmap='gray',vmin=0,vmax=255)
                plt.pause(0.00000001)

        self.pano_frame = self.redo_pano_frame