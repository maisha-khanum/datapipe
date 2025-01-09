from math import trunc
import bz2,gzip
import time
import pickle,json
import rosbag
import yaml
import argparse
from tqdm import tqdm
from numba import jit
import numpy as np
import ros_numpy as rnp
import matplotlib.pyplot as plt
from std_msgs.msg import Int32, String
import sensor_msgs.point_cloud2 as pc2  # api of pc2
from sensor_msgs.msg import PointCloud2, Image, JointState # This is message definition of pc2
import sensor_msgs
from scipy.spatial.transform import Rotation as R
# from cv_bridge import CvBridge

# Multi process
from joblib import Parallel, delayed
import multiprocessing

import sys
HZ = 20

# Note:
# D1. Speed up
# D2. Add variance
# D3. panorama mod
#   - window data, physical distance sphere key point filter.

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
def normalise_to_255(a):
    # return (((a - min(a)) / float((max(a) - min(a))*0+10)) * 255.0).astype(np.uint8)
    return (a/10.0*255.0).astype(np.uint8)

# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# This is updated to align with cpp code, do not trust the py code for now.
def generate_pano(pc_frame_glob, curr_pos,r,view_dist=10.0,filter=False):        
    total_pt_array = np.vstack(pc_frame_glob)

    # r = R.from_quat(curr_quat).as_matrix()
    total_pt_array[:,:3] = total_pt_array[:,:3]-curr_pos
    total_pt_array[:,:3] = total_pt_array[:,:3].dot(r) # rotate to user heading

    x = total_pt_array[:,0]
    y = total_pt_array[:,1]
    z = total_pt_array[:,2]
    # x,y,z = total_pt_array.T

    # r = total_pt_array[:,3]
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

    # d = d # normalize to 10 meters in image

    width  = int(SURROUND_U_MAX - SURROUND_U_MIN)
    height = int(SURROUND_V_MAX - SURROUND_V_MIN)
    # surround     = np.zeros((height, width, 3), dtype=np.float32)
    surround_img = np.zeros((height, width), dtype=np.uint8)+255

    # surround[v, u, 0] = d    # still want to return the true depth
    # surround[v, u, 1] = d
    # surround[v, u, 2] = d

    @jit(nopython=True) # OPTIMIZED
    def proj_and_clip(tmp, u,v,d,height, width): # OPTIMIZED
        for i in range(len(u)):
            if tmp[v[i],u[i]] > d[i]:
                tmp[v[i],u[i]] = d[i]
            # tmp[v[i],u[i]] = min(tmp[v[i],u[i]],d[i])

    # for i in range(0, len(v)):
    #     if v[i] > 170:
    #         print(u[i],v[i],dist[i],dxy[i],x[i],y[i],z[i])
    proj_and_clip(surround_img, u,v,normalise_to_255(np.clip(dist,     0.0, 10.0)),height, width)

    # print('===============================')
    # print(surround_img[178,:])

    # surround_img[v, u, 0] = normalise_to_255(np.clip(d,     0.0, 10.0))
    # surround_img[v, u, 1] = surround_img[v, u, 0] # grey scale
    # surround_img[v, u, 2] = surround_img[v, u, 0] # grey scale
    
    # surround_img[v, u, 1] = normalise_to_255(np.clip(0*z+1.8, 0, 100))
    # surround_img[v, u, 2] = normalise_to_255(np.clip(r,     0, 30))

    # # Crop the image to 60 - 60
    # cutidx = int(180/SURROUND_V_STEP/6)
    # surround_img = surround_img[cutidx:-cutidx,:,:]

    return surround_img

def generate_pano_old(pc_frame_glob, curr_pos,r):        
    total_pt_array = np.vstack(pc_frame_glob)
    # r = R.from_quat(curr_quat).as_matrix()
    total_pt_array[:,:3] = total_pt_array[:,:3]-curr_pos
    total_pt_array[:,:3] = total_pt_array[:,:3].dot(r) # rotate to user heading

    x = total_pt_array[:,0]
    y = total_pt_array[:,1]
    z = total_pt_array[:,2]
    # r = total_pt_array[:,3]
    d = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin

    u,v = lidar_to_surround_coords(x,y,z,d)

    # A little trick to make sure we don't see through walls
    rank_idx = np.argsort(d)[::-1] # find index in descending order
    # sort all 3 in descending order by d
    u = u[rank_idx]
    v = v[rank_idx]
    d = d[rank_idx]

    # d = d # normalize to 10 meters in image

    width  = int(SURROUND_U_MAX - SURROUND_U_MIN + 1)
    height = int(SURROUND_V_MAX - SURROUND_V_MIN + 1)
    # surround     = np.zeros((height, width, 3), dtype=np.float32)
    surround_img = np.zeros((height, width, 3), dtype=np.uint8)+255

    # surround[v, u, 0] = d    # still want to return the true depth
    # surround[v, u, 1] = d
    # surround[v, u, 2] = d
    surround_img[v, u, 0] = normalise_to_255(np.clip(d,     0.0, 10.0))
    surround_img[v, u, 1] = surround_img[v, u, 0] # grey scale
    surround_img[v, u, 2] = surround_img[v, u, 0] # grey scale
    
    # surround_img[v, u, 1] = normalise_to_255(np.clip(0*z+1.8, 0, 100))
    # surround_img[v, u, 2] = normalise_to_255(np.clip(r,     0, 30))

    # Crop the image to 60 - 60
    cutidx = int(180/SURROUND_V_STEP/6)
    surround_img = surround_img[cutidx:-cutidx,:,:]

    return surround_img

# ========================
# Helper classes
# ========================

class training_set_generator:
    def __init__(self, fpath, save_video=False, parallel=True, view_dist=10.0, window_sz=64):
        self.fpath      = fpath
        self.bag        = rosbag.Bag(self.fpath)
        self.info_dict  = yaml.safe_load(self.bag._get_yaml_info())
        self.par        = parallel           # parallel flag
        self.save_video = save_video
        print(self.info_dict.keys())

        # Panorama parameters
        self.view_dist = view_dist
        self.window_sz = window_sz

        # extracted data
        self.video_t       = []   # time of each  [camera] frame
        self.video_frame   = []   # raw frames of [camera] stream
        self.pano_t        = []   # time of each  [panorama] frame
        self.pano_frame    = []   # raw frames of [panorama] stream
        self.cpano_frame   = []   # crystal ball frames of [panorama] stream
        self.pc_t          = []   # time of each  [point cloud] frame
        self.pc_frame      = []   # raw frames of [point cloud] stream
        self.pc_frame_glob = []   # global frames of [point cloud] stream
        self.data_array    = []   # Other data, sorted by time

        # temp data
        self.temp_pos     = [0,0,0]
        self.temp_pose    = [0,0,0,1] # make sure we don't have zero length quaternion
        self.temp_pose_var= [0]*3 # Updated: only care about lower triangle diag
        self.temp_vel     = [0,0,0]
        self.temp_ang_vel = [0,0,0]
        self.temp_step    = 0
        self.temp_joint   = [0,0,0,0]
        self.pc_idx       = 0
        self.video_idx    = 0
        self.pano_idx     = 0
        self.valid_start  = 0

        self.data_line = [0]*22

    def init_par(self):
        self.num_cores = multiprocessing.cpu_count()

    # Clean up function
    def exit(self):
        self.bag.close()

    # DTODO: test run the system to see if there is any jump, and try to identify jumps
    # Done: Check Edgar Kraft UKF covariance calculation
    # Save training set to pickle file
    def save(self,save_fpath):
        import joblib
        print('[INFO] Saving to '+save_fpath)
        t_start = time.time()
        if len(self.cpano_frame)>1:
            # Done: Clean this up, might not need: video_t(confirmed), video_frame(confirmed), pano_frame, pc_frame(confirmed)
            # Space: Video: 1.3 G
            #        point cloud:450 mb
            #        the rest: 46 mb
            cache_dict = {"pano_t": self.pano_t, "pano_frame": self.pano_frame, "cyst_pano_frame": self.cpano_frame,\
                        # "video_t": self.video_t, "video_frame": self.video_frame,\
                        # "pc_t": self.pc_t, "pc_frame": self.pc_frame,\
                        "data_array": self.data_array}
                        # "data_array": self.data_array[self.valid_start:]} # cut the index before all 3 start streaming
        else:
            cache_dict = {"pano_t": self.pano_t, "pano_frame": self.pano_frame,\
                        "video_t": self.video_t, "video_frame": self.video_frame,\
                        # "pc_t": self.pc_t, "pc_frame": self.pc_frame,\
                        "data_array": self.data_array}
                        # "data_array": self.data_array[self.valid_start:]} # cut the index before all 3 start streaming
        
        # Write to file
        # outfile = open(save_fpath,'wb')           # Fastest 57s 4300MB
        # outfile = bz2.BZ2File(save_fpath, 'w')    # Least space 473s 341MB
        # outfile = gzip.open(save_fpath,'wb')      # most economical 310s 493MB
        # pickle.dump(cache_dict,outfile)
        # outfile.close()
        joblib.dump(cache_dict, save_fpath, compress=('lz4', 1))
        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))

    # Generate training set and wash data
    def generate_set(self, calib_fac):
        # for each message, populate the training set array
        # with tqdm(total=self.bag.get_message_count()) as pbar:
        for topic, msg, t in self.bag.read_messages(): # topics=['/step_counter','/rosout']
            self.extract_data(topic,msg,t, calib_fac)
                # pbar.update(1)

        # Re sample the data_array to form a fixed time_step data
        # self.resample()

        self.data_array = np.array(self.data_array,dtype=float)

    def resample(self):
        print('[INFO] Resampling data')
        print(type(self.data_array),len(self.data_array),self.data_array.shape)
        print(self.valid_start,self.data_array[self.valid_start][0])
        self.data_array = self.data_array[self.valid_start:]
        
        data_array_t = np.array(self.data_array)[:,0]
        t_start = self.data_array[0][0]
        t_end   = self.data_array[-1][0]
        t_step  = 1.0/HZ # 10 hz instead of 20
        mode    = 'nearest' # or 'interpolate'
        print(t_start,t_end)

        resampled_data_array = []
        # NEW: do not remove the first 5 sec, do that in the data loader.
        for t in tqdm(np.arange(t_start,t_end+0.5*t_step-1,t_step)): # remove the last 1 second in the dataset and first 5 second
            idx = np.argmin(abs(data_array_t - t))
            if mode == 'nearest':
                resampled_line = [t]+list(self.data_array[idx][1:])
            else:
                pass
                # t_old = self.data_array[idx,0]
                # if t_old > t:
                #     t_1 = 
                #     t_2 = t_old

            resampled_data_array.append(resampled_line)
        print('[INFO] Sample size',len(resampled_data_array))
        self.data_array = np.array(resampled_data_array)

    ##### Done: Find variance of pose #####
    def generate_var_pose(self,window_sz=1000): # 1000 ~ 3.5s
        print("[INFO] Calculating variance of pose")
        print("data_array size",self.data_array.shape)
        if self.data_array.shape[0] <= 2:
            print('[WARN] Not enough data to calculate variance')
            return

        for i in tqdm(range(2,self.data_array.shape[0])):
            window = self.data_array[max(0,i-window_sz):i,1:8]
            pos,quat = window[:,0:3],window[:,3:]
            r = R.from_quat(quat)
            r2torso = R.from_rotvec([0,-0.355,0]) 
            r_torso = r*r2torso
            rot_array = r_torso.as_matrix()
            ori_vec_x =  rot_array[:,[2],1]
            ori_vec_y =  rot_array[:,[0],2]
            ori_vec_z = -rot_array[:,[0],1]
            # convert from [position(3) quat(4)] -> [position(3) rot_vec(3)]
            # rotvec_window = np.hstack([pos,r.as_rotvec()])
            rotvec_window = np.hstack([pos,ori_vec_x,ori_vec_y,ori_vec_z])
            
            cov = np.cov(rotvec_window,rowvar=False).T
            # save only top 6 and bottom 6 to data dict
            x,y = np.triu_indices(3)
            # pose_var = np.hstack([cov[x,y],cov[x+3,y+3]])
            pose_var = np.array([cov[3,3],cov[4,4],cov[5,5]])
            self.data_array[i,8:8+3] = pose_var


    # Generate panorama that peeks into the future. Must call after generate_set()
    def generate_crystal_ball_panorama(self):
        # Input check
        if len(self.data_array)<5:
            print("[WARNING] You must call generate_set() before generate_crystal_ball_panorama()")
            return
        
        # Convert every pc into global coordinate
        key_poses = []
        with tqdm(total=len(self.pc_frame)) as pbar:
            for idx, pc in enumerate(self.pc_frame):
                # print('[INFO] Processing point cloud {}/{}'.format(idx+1,len(self.pc_frame)))
                # pbar.write('fefefeee')
                curr_pos,curr_quat = self.find_nearest_pos(self.pc_t[idx])
                key_poses.append(np.append(curr_pos,curr_quat))
                pc_global = self.pc2globalpc(pc, curr_pos,curr_quat)
                self.pc_frame_glob.append(pc_global)
                # time.sleep(0.1)
                pbar.update(1)

        # Collect all panorama in a readable format
        # total_pc_array = np.concatenate(self.pc_frame_glob,axis=0)

        # Generate for each key position in the path
        n_keypose = len(key_poses)
        self.cpano_frame = [None]*n_keypose

        def make_pano_single_pass(idx,key_pose):
            print('[INFO] Generating Panorama for pose {}/{}'.format(idx+1,n_keypose))
            print('[INFO] Keypose location:',key_pose[:3])
            t1 = time.time()
            r = R.from_quat(key_pose[3:]).as_matrix()
            #####  Only input window data #####
            window_l, window_r = max(idx-self.window_sz+1,0), idx+1
            panorama = generate_pano(self.pc_frame_glob[window_l:window_r], key_pose[:3],r,view_dist=self.view_dist,filter=True)
            t2 = time.time()
            print("Taken",t2-t1)
            return panorama

        if self.par:
            self.init_par()
            print(self.num_cores)
            print(12)
            result = Parallel(n_jobs=7,backend="threading")(delayed(make_pano_single_pass)(i,key_poses[i]) for i in range(n_keypose))
            self.cpano_frame = result
        else:
            for idx in range(len(key_poses)):
                panorama = make_pano_single_pass(idx,key_poses[idx])
                self.cpano_frame[idx] = panorama

                plt.clf()
                plt.imshow(panorama,cmap='gray',vmin=0,vmax=255)
                plt.pause(0.00000001)
    
    # For each time stamp entry in key pose list, find the corresponding pose in raw data
    def find_nearest_pos(self,t):
        idx = np.argmin(abs(self.data_array[:,0] - t))
        return self.data_array[idx,1:4].copy(), self.data_array[idx,4:8].copy()
    
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

    def extract_data(self,topic,msg,t, calib_fac):
        new_entry = False
        if (topic == '/d400/color/image_raw'):
            # print('[TYPE] video_frame')
            # pbar.write('[TYPE] video_frame')
            if self.save_video:
                # Convert to cv frame
                msg.__class__ = sensor_msgs.msg._Image.Image
                img = rnp.numpify(msg)
                
                # Gather info from header
                msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
                print("video: bag_time",t,"msg_time",msg_t,"diff",t.to_sec()-msg_t)
                # time.sleep(0.5)
                # Save to respective array
                self.video_t.append(msg_t)
                self.video_frame.append(img)
                self.video_idx = len(self.video_t)-1
                new_entry = True

                # update cut point
                if self.video_idx == 0:
                    self.valid_start = max(len(self.data_array),self.valid_start)

                # print('[DEBUG] ',msg.height,msg.width)

                # Draw frame
                # plt.clf()
                # plt.imshow(img)
                # plt.pause(0.000001)

        elif (topic == '/rosout'):
            pass
            # print('[TYPE] Console message')
            # pbar.write('[TYPE] Console message')

        elif (topic == '/rosout_agg'):
            # print('[TYPE] Aggregated Console Message')
            # pbar.write('[TYPE] Aggregated Console Message')
            msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
            # print('[DEBUG] ',msg_t,msg.msg)

        elif (topic == '/save_pc'):
            # print('[TYPE] point cloud message')
            # pbar.write('[TYPE] point cloud message')
            msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
            print("PC: bag_time",t,"msg_time",msg_t,"diff",t.to_sec()-msg_t)
            # time.sleep(0.5)
            # convert from pc2 to numpy array
            # pc_gen   = pc2.read_points(msg, skip_nans=True) 
            # pt_array = np.array(list(pc_gen))
            pt_array = rnp.point_cloud2.pointcloud2_to_xyz_array(msg,remove_nans=True) # OPTIMIZED
            pt_array = pt_array.reshape(-1,3)
            pt_array = pt_array[np.sum(pt_array,axis=1)>0.3,:] # remove all zero points

            pt_array[:,[0,1,2]] = pt_array[:,[0,1,2]] * calib_fac  # NEW: calibrate the point cloud from source
            d = pt_array[:,0]**2 + pt_array[:,1]**2 + pt_array[:,2]**2
            pt_array = pt_array[d<self.view_dist**2]
            print(pt_array.shape)

            # Save to respective array
            self.pc_t.append(msg_t)
            self.pc_frame.append(pt_array)
            self.pc_idx = len(self.pc_t)-1
            new_entry = True

            # update cut point
            if self.pc_idx == 0:
                self.valid_start = max(len(self.data_array),self.valid_start)
            
        elif (topic == '/step_counter'):
            # print('[TYPE] Step count')
            if self.temp_step != msg.data:
                self.temp_step = msg.data
                msg_t = t.to_sec()
                new_entry = True

        elif (topic == '/t265/odom/sample'):
            if msg.child_frame_id == 't265_pose_frame':
                # print('[TYPE] Pose')
                msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
                self.temp_pos  = [msg.pose.pose.position.x,    msg.pose.pose.position.y,    msg.pose.pose.position.z]
                self.temp_pose = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,msg.pose.pose.orientation.w]
                # should we also add in the velocity? YES
                self.temp_vel     = [msg.twist.twist.linear.x,  msg.twist.twist.linear.y,  msg.twist.twist.linear.z]
                self.temp_ang_vel = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                # print(msg)
                new_entry = True
            
        elif (topic == '/testpano'):
            # print('[TYPE] Panorama Image')
            # Convert to cv frame
            # bridge = CvBridge()
            msg.__class__ = sensor_msgs.msg._Image.Image
            img = rnp.numpify(msg)
            # img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Gather info from header
            msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
            # print("Pano: bag_time",t,"msg_time",msg_t,"diff",t.to_sec()-msg_t)
            # time.sleep(0.5)
            if msg_t < 1.0:
                msg_t = t.to_sec()

            # Save to respective array
            self.pano_t.append(msg_t)
            self.pano_frame.append(img)
            self.pano_idx = len(self.pano_t)-1
            new_entry = True

            # update cut point
            if self.pano_idx == 0:
                self.valid_start = max(len(self.data_array),self.valid_start)


            # print('[DEBUG] ',msg.height,msg.width)

            # Draw panorama
            # plt.clf()
            # plt.imshow(img)
            # plt.pause(0.000001)

        elif (topic == '/tf'):
            # print('[TYPE] TF data')
            pass

        elif (topic == '/tf_static'):
            # print('[TYPE] TF data static')
            pass

        elif (topic == '/joint_states'):
            # print('[TYPE] Joint State')
            msg_t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
            # print("Joint: bag_time",t,"msg_time",msg_t,"diff",t.to_sec()-msg_t)
            # time.sleep(0.5)
            self.temp_joint = list(msg.position) # [LB RB LK RK]
            new_entry = True

        else:
            print('[ALERT] Ununsed',topic,type(msg),t,)

        # construct line of entry and save to data_array
        # Data_line format: [ time, position(3), pose_quat(4), pose_variance(2*6)
        #                     velocity(3), angular_velocity_euler(3), 
        #                     step(1), joint_angles(4), pc_index(1), 
        #                     pano_index(1), camera_index(1) ]
        if new_entry:
            old_line = self.data_line[:] # deep copy old data entry
            if msg_t < old_line[0]:
                print('[ALERT] Time is not monotonically increasing')
                time.sleep(1)
            # Note: self.temp_pose_var is a dummy value here, it keeps the 3 position open for
            #       generate_var_pose() to fill in later.
            self.data_line = [msg_t] + self.temp_pos + self.temp_pose + self.temp_pose_var + \
                             self.temp_vel + self.temp_ang_vel + \
                             [self.temp_step] + self.temp_joint + \
                             [self.pc_idx, self.pano_idx, self.video_idx]
            # print('[INFO] Adding new data entry')
            
            # Only append line if new entry is not a repetition of old line
            if self.data_line[1:] != old_line[1:]:
                self.data_array.append(self.data_line[:])
            else:
                # if there is a repetition, means new entry is activated, but no new data.
                # self.valid_start = len(self.data_array)
                print('[WARNING] repetitive entry found, skipping')
                print(topic,old_line,self.data_line)
                # time.sleep(0.5)

    # Redo the panorama generation with calibration parameters. It will replace the current panorama with
    # new panorama for the model training data.
    def redo_panorama(self,calib_fac,sample_ratio=0.07):
        # Input check
        if len(self.data_array)<5:
            print("[WARNING] You must call generate_set() before redo_panorama()")
            return
        
        # Convert every pc into global coordinate
        # key_poses = []
        with tqdm(total=len(self.pc_frame)) as pbar:
            for idx, pc in enumerate(self.pc_frame):
                # print('[INFO] Processing point cloud {}/{}'.format(idx+1,len(self.pc_frame)))
                # pbar.write('fefefeee')
                curr_pos,curr_quat = self.find_nearest_pos(self.pc_t[idx])
                # key_poses.append(np.append(curr_pos,curr_quat))

                # downsample and scale point cloud
                calib_scale = calib_fac
                n_pts = int(pc.shape[0]*sample_ratio)
                sample_idx = np.random.permutation(pc.shape[0])[:n_pts]
                pc_post = pc[sample_idx,:] #*calib_scale   # NEW: since we already mulitplied it in the raw pc

                pc_global = self.pc2globalpc(pc_post, curr_pos,curr_quat)
                self.pc_frame_glob.append(pc_global)
                # time.sleep(0.1)
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

            # ========This is not correct======
            # r = R.from_quat(key_pose[3:]).as_rotvec()
            # r = R.from_rotvec([0,0,r[2]]).as_matrix() # changed to z rotation only

            # New way, gopro like horizon locked
            r = R.from_quat(key_pose[3:]).as_euler('ZYX')
            r = R.from_euler('ZYX',[r[0],0,0]).as_matrix()

            idx = sum(np.array(self.pc_t)<pano_t)                                 # most up to date point cloud index
            window_l, window_r = max(idx-self.window_sz+1,0), idx+1     # Only input window data
            print(window_l,window_r)
            panorama = generate_pano(self.pc_frame_glob[window_l:window_r], key_pose[:3],r,view_dist=self.view_dist,filter=True)
            t2 = time.time()
            print("Taken",t2-t1)
            return panorama

        print(self.par)
        if self.par:
            self.init_par()
            print(self.num_cores)
            result = Parallel(n_jobs=7,backend="threading")(delayed(make_pano_single_pass)(self.pano_t[i]) for i in range(n_keypose))
            self.redo_pano_frame = result
        else:
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
        print(len(self.redo_pano_frame),len(self.pano_frame))

    def update_gait(self):
        steps = self.data_array[:,17]
        joint_angles = self.data_array[:,18] # Take IMU0 as reference
        window_sz = int(2.5*HZ) # 10HZ
        joint_mean = 0
        counts = np.zeros(joint_angles.shape)
        print(max(steps))

        for i in range(0,joint_angles.shape[0]-1):
            joint_mean = (joint_mean*i + joint_angles[i])/(i+1) # Running mean
            if joint_angles[i] < joint_mean < joint_angles[i+1]:
                counts[i] += 1
            if joint_angles[i+1] < joint_mean < joint_angles[i]:
                counts[i] += 1
        print(sum(counts))

        for i in range(len(counts)):
            freq = np.sum(counts[max(0,i-window_sz):i+1])/(window_sz/HZ)
            # print(i,freq)
            self.data_array[i,17] = freq




# ========================
# Main logic
# ========================
if __name__ == "__main__":
    # fpath = '../bag_recording/field_2021-07-16-11-46-47.bag'
    # fpath = '../bag_recording/field_2021-08-18-15-59-25.bag'
    # fpath = '../bag_recording/field_2021-10-17-15-58-05.bag'
    # fpath = '../bag_recording/field_2021-12-09-16-45-58.bag.active'
    # fpath = '/media/askker/Extreme Pro/bag_recording/field_2021-12-09-16-53-24.bag'

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath',      type=str,   default='??')
    parser.add_argument('--save_video', type=bool,  default=False)
    parser.add_argument('--no_parallel',   action='store_true',  default=False)
    parser.add_argument('--view_dist',  type=float, default=10.0)
    parser.add_argument('--calib_fac',  type=float, default=0.72) # sometimes 0.82 is good
    parser.add_argument('--window_sz',  type=int,   default=32)
    args = parser.parse_args()

    # save_fpath = '/media/askker/Extreme SSD/SmartBelt/processed_training_setV3/20Hz_'+args.fpath.split('/')[-1].split('.')[0] # extract the fname
    # save_fpath = '/'.join(args.fpath.split('/')[:-1]+['20HZVZ_'+args.fpath.split('/')[-1].split('.')[0]])
    save_fpathS = '/'.join(args.fpath.split('/')[:-1]+['20HZVZS_'+args.fpath.split('/')[-1].split('.')[0]])

    # (V)10HZVZS: (Video) + 10 Hz + Variance + Z rot only + step

    generator = training_set_generator(args.fpath,
                                       save_video=args.save_video, 
                                       parallel=(not args.no_parallel), 
                                       view_dist=args.view_dist, 
                                       window_sz=args.window_sz)
    generator.generate_set(calib_fac=args.calib_fac)
    # generator.generate_crystal_ball_panorama() # run this to add cpano entry in data dict

    # for V2DataRedo sample ratio = 0.3, for old dataset we use 0.2
    # generator.redo_panorama(calib_fac=args.calib_fac, sample_ratio=0.3)                  # !! Only run this on the old dataset that contains uncalibrated point cloud !!
    generator.generate_var_pose()              # run this to fill in the 3 variance in self.data_array
    generator.resample()                       # run this to resample the data_array to 20Hz   
    # generator.save(save_fpath)
    generator.update_gait()                    # !! Only run this to replace/update the gait entry in data dict using IMU joint angles
    generator.save(save_fpathS)

    # Clean up
    plt.show()
    generator.exit()

# ========================
# Note on bag recordings
# ========================
# 07/16-11:46 first usable bag recording, probably missing some functions and not much motion
# 08/18-15:59 most tested bag recording in development, taken at home, short, 46 s usable data, no leg motion
# 10/17-15:30 Full leg motion (Rosbag + screen recording) laggy (first test run of the day)
# 10/17-15:44 Full leg motion (Rosbag + screen recording) laggy
# -=10/17-15:54=- Full leg motion (screen only) not laggy
# 10/17-15:49 Full leg motion (Rosbag only) not laggy
# 10/17-15:58 long walk around the lab (Rosbag only) (outside) not laggy