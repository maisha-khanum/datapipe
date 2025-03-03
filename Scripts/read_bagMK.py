import numpy as np
from rosbags.rosbag2 import Reader
from tqdm import tqdm
import joblib
import time

# Constants
HZ = 20  # Resample to 20 Hz
BAG_PATH = '/home/mkhanum/datapipe/Bags/stair1'
TOPICS = {"rgb": "/d455/color/image_raw", "pc": "/d455/depth/color/points"}

# Step 1: Extract timestamps
timestamps = {key: [] for key in TOPICS}

with Reader(BAG_PATH) as reader:
    for conn in reader.connections:
        for key, topic in TOPICS.items():
            if conn.topic == topic:
                for _, timestamp, _ in reader.messages(connections=[conn]):
                    timestamps[key].append(timestamp)

# Step 2: Resample and Process Messages One at a Time
resampled_data = {"rgb": [], "pc": []}

for key, topic in TOPICS.items():
    if len(timestamps[key]) > 1:
        timestamps_arr = np.array(timestamps[key])
        t_start, t_end = timestamps_arr[0], timestamps_arr[-1] - (1.0 / HZ)
        t_step = 1.0 / HZ
        new_time_points = np.arange(t_start, t_end, t_step)

        print(f"[INFO] Resampling {topic}")

        with Reader(BAG_PATH) as reader:
            connection = next((conn for conn in reader.connections if conn.topic == topic), None)
            if not connection:
                continue

            message_cache = {}

            for t in tqdm(new_time_points, desc=f"Resampling {key}"):
                idx = np.argmin(abs(timestamps_arr - t))
                nearest_timestamp = timestamps_arr[idx]

                if nearest_timestamp in message_cache:
                    resampled_data[key].append((t, message_cache[nearest_timestamp]))
                else:
                    for _, timestamp, rawdata in reader.messages(connections=[connection]):
                        if timestamp == nearest_timestamp:
                            resampled_data[key].append((t, rawdata))
                            message_cache[nearest_timestamp] = rawdata
                            break

        print(f"[INFO] Resampled {len(resampled_data[key])} messages for {topic}")

# Step 3: Save Data

def save_data(resampled_data, save_fpath):
    print('[INFO] Saving to ' + save_fpath)
    t_start = time.time()
    joblib.dump(resampled_data, save_fpath, compress=('lz4', 1))
    print('[INFO] Done saving, took {:.2f}s'.format(time.time() - t_start))

save_data(resampled_data, '/home/mkhanum/datapipe/resampled_data.pkl')

# Clean up memory
del timestamps
del timestamps_arr
