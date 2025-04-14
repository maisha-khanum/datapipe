import matplotlib.pyplot as plt
import pandas as pd

def plot_gsam2_timing(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    # Separate data points based on mask emptiness
    empty_mask_data = df[df['Empty Mask'] == True]
    non_empty_mask_data = df[df['Empty Mask'] == False]

    # Convert duration from milliseconds to seconds
    non_empty_mask_data['Duration (s)'] = non_empty_mask_data['Duration (ms)'] / 1000.0
    empty_mask_data['Duration (s)'] = empty_mask_data['Duration (ms)'] / 1000.0

    plt.scatter(non_empty_mask_data['Message Counter'], non_empty_mask_data['Duration (s)'], color='blue', label='Mask Found')
    plt.scatter(empty_mask_data['Message Counter'], empty_mask_data['Duration (s)'], color='red', label='No Mask Found')

    plt.xlabel('Message #')
    plt.ylabel('Duration (s)')  # Updated y-axis label
    plt.title('GSAM2 Duration vs. Message Count')
    plt.legend()
    plt.grid(False)
    plt.show()

# Replace 'gsam2_timing_data.csv' with your actual file name
plot_gsam2_timing('/home/mkhanum/datapipe/ros2_ws/gsam2_timing_data.csv')