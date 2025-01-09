import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import time
import numpy as np
from segment_anything import build_sam, SamPredictor 
import torch
import pickle, joblib
from typing import Tuple
from tqdm import tqdm

import sys
GSAM_path = '/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/Grounded-Segment-Anything'
sys.path.insert(1, GSAM_path)

SCALE_FACTOR = 1.5  # Adjust this value as needed for screen resolution

def alpha_blend(original_image, mask, color, alpha_value):
    color_image = np.ones(original_image.shape, dtype=original_image.dtype) * np.array(color)
    alpha_channel = mask * alpha_value
    blended_image = alpha_channel[..., None] * color_image + (1 - alpha_channel[..., None]) * original_image
    return blended_image.astype(original_image.dtype)

class SAM_Data_Wrapper:
    def __init__(self):
        self.load_SAM()
        self.load_data()
        pass

    def load_SAM(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ==============Load SAM================
        print('\nLoading SAM')
        GSAM_path = '/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/Grounded-Segment-Anything'
        sam_checkpoint = GSAM_path + '/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)
        print('SAM loaded\n')

    def load_data(self):
        fpath = '/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/'+'selected_video_frames'
        print("\nLoading "+fpath)
        infile = open(fpath,'rb')
        self.frames = pickle.load(infile, encoding='latin1')
        infile.close()
        selected_idx = np.random.permutation(np.arange(len(self.frames)))[:10]
        self.frames = [self.frames[i] for i in selected_idx]

class Segmentation_frame:
    def __init__(self, image, seg_engine, n_ch=6):
        # Data
        self.original_image = image  # np version
        self.point_lst = [[] for i in range(n_ch)]
        self.mask_idx_lst = [[] for i in range(n_ch)]
        self.channel_masks = np.zeros((480, 640, n_ch))

        # Options
        self.seg_engine = seg_engine
        self.n_ch = n_ch            # Number of channels
        self.curr_ch = 0       # Current channel
        
        # Set access point
        self.image = Image.fromarray(self.original_image) # .resize((self.new_width, self.new_height))

    def update_image(self, new_frame=False):
        if new_frame:
            t1 = time.time()
            self.seg_engine.sam.set_image(self.original_image)
            t2 = time.time()
            print(f"Time taken to set image: {t2-t1:.2f} seconds")

        # gather relevant info
        curr_ch = self.curr_ch
        point_lst = np.array(self.point_lst[curr_ch])

        if len(point_lst) > 0:
            self.update_ch_mask()
            masked_image = self.original_image
            masked_image = alpha_blend(masked_image, self.channel_masks[:,:,curr_ch], [0, 255, 0], 0.3)
            print(masked_image.shape)
            self.image = Image.fromarray(masked_image)
        else:
            self.image = Image.fromarray(self.original_image)

    # update the mask for the current channel according to the points
    def update_ch_mask(self):
        t1 = time.time()
        point_lst = self.point_lst[self.curr_ch]
        mask_idx_lst = self.mask_idx_lst[self.curr_ch]
        total_masks = np.zeros((480, 640))
        for i in range(len(point_lst)):
            input_points = np.array([point_lst[i]])
            input_labels = np.array([1])
            masks, scores, logits = self.seg_engine.sam.predict(point_coords=input_points,
                                                                point_labels=input_labels,
                                                                multimask_output=True
                                                                )
            # print(np.sum(np.sum(masks,axis=1),axis=1))
            masks = masks[min(mask_idx_lst[i], len(masks))] # only take the smallest mask
            total_masks += masks
        t2 = time.time()
        self.channel_masks[:,:,self.curr_ch] = total_masks
        print(f"Prediction took: {t2-t1:.2f} s")

        self.channel_masks = np.clip(self.channel_masks, 0, 1) # resulting value either 0 or 1



class Tool_GUI:
    def __init__(self, root):
        # Variables and inits
        self.root = root
        self.add_mode = True
        self.dot_radius = 5
        self.mask_idx = 0

            # Segmentor model definition
        self.seg_engine = SAM_Data_Wrapper()

            # Segmentation frame definition
        self.current_seg_index = 0
        self.seg_lst = []
        for i in range(len(self.seg_engine.frames)):
            self.seg_lst.append(Segmentation_frame(self.seg_engine.frames[i], self.seg_engine))
        self.curr_frame = self.seg_lst[0]


        # Layout Definition
        self.image_frame = tk.Frame(root, bg='white')
        self.image_frame.grid(row=0, column=0, sticky="nsew")

            # Image
        self.new_width = int(640 * SCALE_FACTOR)
        self.new_height = int(480 * SCALE_FACTOR)
        self.label = tk.Label(self.image_frame, image=ImageTk.PhotoImage(self.curr_frame.image), bg='white', bd=10*SCALE_FACTOR)
        self.label.grid(row=0, column=0, padx=10*SCALE_FACTOR, pady=15*SCALE_FACTOR)
        self.label.bind("<Button-1>", self.on_image_click)
        self.update_image(new_frame=True)

            # tool column on the right
        self.right_frame = tk.Frame(root, bg='white', width=500*SCALE_FACTOR)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10*SCALE_FACTOR)

                # Prev/Next Frame
        self.prev_next_frame = tk.Frame(self.right_frame, bg='white')
        self.prev_next_frame.pack(padx=10*SCALE_FACTOR, pady=20*SCALE_FACTOR)

                    # Next image Button
        font_size = int(10 * SCALE_FACTOR)
        self.prev_button = tk.Button(self.prev_next_frame, text="Prev Frame", command=lambda: self.load_frame(-1), font=("Arial", font_size))
        self.prev_button.grid(row=0, column=0, padx=5*SCALE_FACTOR)

        self.next_button = tk.Button(self.prev_next_frame, text="Next Frame", command=lambda: self.load_frame(1), font=("Arial", font_size))
        self.next_button.grid(row=0, column=1, padx=5*SCALE_FACTOR)

                    # Progress bar
        self.progress = ttk.Progressbar(self.prev_next_frame, orient="horizontal", length=300*SCALE_FACTOR, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=2, pady=10*SCALE_FACTOR)
        self.progress["maximum"] = len(self.seg_engine.frames) - 1  # -1 since indexing starts from 0
        self.progress["value"] = self.current_seg_index

                # Point Mode frame (inner frame for "Add Point" and "Remove Point" buttons)
        self.point_mode_frame = tk.Frame(self.right_frame, bg='white')
        self.point_mode_frame.pack(pady=5*SCALE_FACTOR)

                    # Text description for "Point Mode"
        self.point_mode_label = tk.Label(self.point_mode_frame, text="Point Mode", font=("Arial", font_size), bg='white')
        self.point_mode_label.grid(row=0, column=0, padx=5*SCALE_FACTOR)

                    # Add Point Button
        self.add_button = tk.Button(self.point_mode_frame, text="Add Point", command=self.add_point, font=("Arial", font_size), relief=tk.SUNKEN)
        self.add_button.grid(row=0, column=1, padx=5*SCALE_FACTOR)

                    # Remove Point Button
        self.remove_button = tk.Button(self.point_mode_frame, text="Remove Point", command=self.remove_point, font=("Arial", font_size))
        self.remove_button.grid(row=0, column=2, padx=5*SCALE_FACTOR)

                # Channels frame (inner frame for "C1" to "C6" buttons)
        channels_frame = tk.Frame(self.right_frame, bg='white')
        channels_frame.pack(pady=5*SCALE_FACTOR)

                    # Text description for "Channels"
        channels_label = tk.Label(channels_frame, text="Channels", font=("Arial", font_size), bg='white')
        channels_label.grid(row=0, column=0, padx=5*SCALE_FACTOR)

                    # Create buttons C1 to C6
        self.channel_buttons = []  # List to store the channel buttons
        for idx in range(6):
            btn = tk.Button(channels_frame, text=f"C{idx}", font=("Arial", font_size), 
                            command=lambda idx=idx: self.channel_button_callback(idx), relief=tk.SUNKEN if idx==0 else tk.RAISED)
            btn.grid(row=0, column=idx+1, padx=5*SCALE_FACTOR)
            self.channel_buttons.append(btn)

        # Mask Size frame (inner frame for "S", "M", "L" buttons)
        mask_size_frame = tk.Frame(self.right_frame, bg='white')
        mask_size_frame.pack(pady=5*SCALE_FACTOR)

        # Text description for "Mask Size"
        mask_size_label = tk.Label(mask_size_frame, text="Mask Size", font=("Arial", font_size), bg='white')
        mask_size_label.grid(row=0, column=0, padx=5*SCALE_FACTOR)

        # Create buttons S, M, L
        mask_sizes = ["S", "M", "L"]
        self.mask_buttons = []  # List to store the mask size buttons
        for idx, size in enumerate(mask_sizes):
            btn = tk.Button(mask_size_frame, text=size, font=("Arial", font_size), 
                            command=lambda idx=idx: self.mask_size_callback(idx), relief=tk.SUNKEN if idx==0 else tk.RAISED)
            btn.grid(row=0, column=idx+1, padx=5*SCALE_FACTOR)
            self.mask_buttons.append(btn)

                # Description Label at the bottom of the right column
        description_text = """
        Ch0: Ground    |   Ch1: Stairs
        Ch2: Doors     |   Ch3: Wall
        Ch4: Obstacles |   Ch5: Movables
        """
        description_label = tk.Label(self.right_frame, text=description_text, font=("Arial", font_size), bg='white')
        description_label.pack(side=tk.BOTTOM, pady=5*SCALE_FACTOR)

            # Save Button
        self.save_button = tk.Button(self.right_frame, text="Save Segmentation", command=self.save, font=("Arial", font_size))
        self.save_button.pack(pady=20*SCALE_FACTOR)

            # Status text box 
        self.status_text = tk.Text(self.right_frame, height=13, width=int(45), wrap=tk.WORD, font=("Arial", font_size))
        self.status_text.insert(tk.END, "Status: Frames loaded.")
        self.status_text.pack(padx=10*SCALE_FACTOR, pady=0*SCALE_FACTOR)

        # Layout Configuration
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_rowconfigure(0, weight=1)

    # ==============Button Callbacks================
    # Callback for the mask size buttons
    def mask_size_callback(self, idx):
        self.mask_idx = idx
        # Toggle button state
        for i in range(len(self.mask_buttons)):
            if i != idx:
                self.mask_buttons[i].config(relief=tk.RAISED)
            else:
                self.mask_buttons[i].config(relief=tk.SUNKEN)


    def channel_button_callback(self, idx, new_frame=False):
        # Toggle button state
        for i in range(len(self.channel_buttons)):
            if i != idx:
                self.channel_buttons[i].config(relief=tk.RAISED)
            else:
                self.channel_buttons[i].config(relief=tk.SUNKEN)
        
        # Store the index (if needed elsewhere)
        self.curr_frame.curr_ch = idx

        self.update_image(new_frame=new_frame)

    def load_frame(self, direction):
        self.current_seg_index += direction
        if self.current_seg_index < 0:
            self.current_seg_index = len(self.seg_lst) - 1
        elif self.current_seg_index >= len(self.seg_lst):
            self.current_seg_index = 0
        self.curr_frame = self.seg_lst[self.current_seg_index]
        self.status_text.insert(tk.END, f"\nSwapped to Frame {self.current_seg_index}")
        self.progress["value"] = self.current_seg_index
        self.channel_button_callback(self.curr_frame.curr_ch, new_frame=True)

    def add_point(self):
        self.add_mode = True
        self.add_button.config(relief=tk.SUNKEN)
        self.remove_button.config(relief=tk.RAISED)

    def remove_point(self):
        self.add_mode = False
        self.remove_button.config(relief=tk.SUNKEN)
        self.add_button.config(relief=tk.RAISED)

    # Get location of the click from image
    def on_image_click(self, event):
        x, y = event.x-10*SCALE_FACTOR, event.y-10*SCALE_FACTOR
        print(f"Clicked at: ({x}, {y})")
        if x < 0 or x > self.new_width or y < 0 or y > self.new_height:
            print("Click registered, but not clicking on image")
            return
        self.status_text.insert(tk.END, f"\nClicked on image at: ({x}, {y})")
        t1 = time.time()

        if self.add_mode:
            # Update state first
            self.curr_frame.point_lst[self.curr_frame.curr_ch].append(np.array([x, y]) / SCALE_FACTOR)
            self.curr_frame.mask_idx_lst[self.curr_frame.curr_ch].append(self.mask_idx)
        else:
            curr_point_lst = np.array(self.curr_frame.point_lst[self.curr_frame.curr_ch])
            if len(curr_point_lst) > 0:
                dist = np.linalg.norm(curr_point_lst - np.array([x, y]) / SCALE_FACTOR, axis=1)
                min_dist_idx = np.argmin(dist)
                if dist[min_dist_idx] < 30:
                    self.curr_frame.point_lst[self.curr_frame.curr_ch].pop(min_dist_idx)
                    self.curr_frame.mask_idx_lst[self.curr_frame.curr_ch].pop(min_dist_idx)
                print(dist)

        self.update_image()
        t2 = time.time()
        print(f"Time taken: {t2-t1:.2f} seconds")

    # ==============Helper Functions================
    # def swap_channel(self):


    def update_image(self, new_frame=False):
        if new_frame:
            # Display the easy version first as we load
            display_image = self.curr_frame.image.copy()
            display_image = display_image.resize((self.new_width, self.new_height))
            tk_image = ImageTk.PhotoImage(display_image)
            self.label.config(image=tk_image)
            self.label.image = tk_image
            self.label.update()
    
        # Modify the image
        self.curr_frame.update_image(new_frame=new_frame)
        display_image = self.curr_frame.image.copy()
        draw = ImageDraw.Draw(display_image)
        for point in self.curr_frame.point_lst[self.curr_frame.curr_ch]:
            draw.ellipse([(point[0]-self.dot_radius, point[1]-self.dot_radius), 
                          (point[0]+self.dot_radius, point[1]+self.dot_radius)], fill='red')
        

        # Update the display
        display_image = display_image.resize((self.new_width, self.new_height))
        tk_image = ImageTk.PhotoImage(display_image)
        # t3 = time.time()
        self.label.config(image=tk_image)
        # t4 = time.time()
        self.label.image = tk_image
        # t5 = time.time()
        # print(f"Time taken to draw: {t4-t3:.2f} seconds")
        # print(f"Time taken to update display: {t5-t4:.2f} seconds")

    def save(self):
        print('Saving segmentation...')
        self.status_text.insert(tk.END, f"\nSaving Segmentation...")

        t_start = time.time()

        save_fpath = '../Training_sets/DinoV2_Manual_finetune'
        data_dict = {}

        # extract data from all segmentation frames
        original_image, point_lst, channel_masks, mask_idx_lst = [], [], [], []
        for i in tqdm(range(len(self.seg_lst))):
            curr_frame = self.seg_lst[i]
            original_image.append(curr_frame.original_image)
            point_lst.append(curr_frame.point_lst)
            mask_idx_lst.append(curr_frame.mask_idx_lst)
            channel_masks.append(curr_frame.channel_masks)

        data_dict['original_image'] = original_image
        data_dict['point_lst']      = point_lst
        data_dict['channel_masks']  = channel_masks
        data_dict['mask_idx_lst']   = mask_idx_lst

        joblib.dump(data_dict, save_fpath, compress=('lz4', 1))

        t_end = time.time()
        time_taken = float(t_end-t_start)
        print('[INFO] Done saving, took {:.2f}s'.format(time_taken))

        self.status_text.insert(tk.END, f"\nDone")

# ==============Helper===============
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

if __name__ == "__main__":
    # image_paths = ['sample1.jpeg', 'sample2.png']
    root = tk.Tk()
    root.title("DinoV2 manual labeling tool")
    root.geometry(str(int(1180*SCALE_FACTOR))+"x"+str(int(540*SCALE_FACTOR)))
    app = Tool_GUI(root)
    root.mainloop()