import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
import numpy as np
import matplotlib.pyplot as plt

def read_closest_points(bag_path, topic_name='/segmented_pointcloud'):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    type_map = reader.get_all_topics_and_types()
    type_dict = {t.name: t.type for t in type_map}
    
    msg_type = get_message(type_dict[topic_name])
    closest_distances = []
    timestamps = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic != topic_name:
            continue
        
        msg = deserialize_message(data, msg_type)
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if not points:
            continue

        # Compute distances and get the minimum
        points_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)
        distances = np.linalg.norm(points_array, axis=1)
        min_distance = np.min(distances)

        closest_distances.append(min_distance)
        timestamps.append(t * 1e-9)  # convert to seconds

    return timestamps, closest_distances

def plot_closest_distances(timestamps, distances):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, distances, label='Closest Point Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Closest Point Distance Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_rosbag> [<topic_name>]")
        exit(1)
    
    bag_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else '/segmented_pointcloud'

    rclpy.init()
    timestamps, distances = read_closest_points(bag_path, topic_name)
    plot_closest_distances(timestamps, distances)
    rclpy.shutdown()
