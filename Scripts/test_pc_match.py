import numpy as np
import pickle
import matplotlib.pyplot as plt
import time 
from skimage.transform import resize

bag_name = 'realsense_2022-08-01-17-25-56_lag.bag'
fpath = '../Training_sets/'+'V20HZVZS_' + bag_name[:-4]
print("loading "+fpath)

infile = open(fpath,'rb')
data_dict = pickle.load(infile, encoding='latin1')
print(data_dict.keys())

video_t = np.array(data_dict['video_t'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig2 = plt.figure()

plot = False
hfov, vfov = 60, 45
for i in range(20,len(data_dict['pc_frame'])):
    # ===============Correspond to video===========
    pc_t = data_dict['pc_t'][i]
    if pc_t > data_dict['data_array'][0,0] - 0.1:
        t1 = time.time()
        pc_frame = data_dict['pc_frame'][i]
        video_idx = np.argmin(abs(video_t-pc_t))    # corresponding video frame index
        data_idx = np.argmin(abs(data_dict['data_array'][:,0]-pc_t))
        curr_video_frame = data_dict['video_frame'][video_idx]   # corresponding video frame
        curr_pos, curr_quat = data_dict['data_array'][data_idx,1:4], data_dict['data_array'][data_idx,4:8]

        # ==================Show frame=================
        if plot:
            # plt.clf()
            # plt.imshow(curr_video_frame)
            # plt.pause(0.1)
            
            rand_idx = np.random.permutation(np.arange(pc_frame.shape[0]))
            pc_frame_d = pc_frame[rand_idx[:int(0.01*rand_idx.shape[0])],:]
            ax.cla()
            # ax.scatter(0,0,0)
            ax.scatter(pc_frame_d[:,0], pc_frame_d[:,1], pc_frame_d[:,2])
            # ax.set_xlim([-10,10])
            # ax.set_ylim([-10,10])
            # ax.set_zlim([-10,10])

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            # z forward, +x right +y down

            fig2.clf()
            plt.imshow(curr_video_frame)
            plt.pause(1)

        # ===============reproject pc to video=========
        curr_pc_frame = np.zeros([480,640,3])
        ver_ang = np.rad2deg(np.arctan2(pc_frame[:,1], pc_frame[:,2]))
        hor_ang = np.rad2deg(np.arctan2(pc_frame[:,0], pc_frame[:,2]))
        # print(max(ver_ang), min(ver_ang))
        # print(max(hor_ang), min(hor_ang),'\n\n\n ')
        depth = pc_frame[:,2]
        
        # for pix in range(pc_frame.shape[0]):
        #     curr_pt = pc_frame[pix,:3]
        #     u,v = int(hor_ang[pix]/hfov*640+320), int(ver_ang[pix]/vfov*480+240)
        #     curr_pc_frame[v,u,:] = depth[pix]/10
        u,v = (hor_ang/hfov*640+320).astype(int), (ver_ang/vfov*480+240).astype(int)
        curr_pc_frame[v,u,:] = depth[:, np.newaxis]/10

        t2 = time.time()
        print(t2-t1)

        # ==============Align and remove black margins==========
        print(min(u),max(u), min(v), max(v))
        curr_pc_frame = curr_pc_frame[2:460, 22:612, :]

        resized_image = resize(curr_pc_frame, (480,618,3), anti_aliasing=True)
        pad_width = ((0, 0), (22, 0), (0, 0))
        # Pad the image with zeros on the left (22 pixels)
        curr_pc_frame = np.pad(resized_image, pad_width=pad_width, mode='constant', constant_values=0)


        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(curr_pc_frame)
        plt.subplot(1,2,2)
        plt.imshow(curr_video_frame)
        plt.pause(100)

            
            
            # print(hor_ang, ver_ang)



print('')
# plt.show()