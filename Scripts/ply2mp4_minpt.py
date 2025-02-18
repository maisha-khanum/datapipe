import open3d as o3d
import cv2
import numpy as np
import os
import re
from tqdm import tqdm

# Ensure Open3D runs in headless mode
os.environ["OPEN3D_RENDERING_DEVICE"] = "EGL"

# Disable WebRTC if available
if hasattr(o3d.visualization.webrtc_server, "disable_webrtc"):
    o3d.visualization.webrtc_server.disable_webrtc()

def natural_sort_key(file_name):
    """ Sort filenames numerically to avoid lexicographic sorting issues. """
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file_name)]

def project_point_to_image(point, intrinsic, width, height):
    """ Projects a 3D point onto a 2D image plane using an intrinsic camera model. """
    fx, fy, cx, cy = intrinsic  # Camera intrinsics (focal lengths and principal point)
    
    if point[2] <= 0:  # Ignore points behind the camera
        return None
    
    u = int((fx * point[0] / point[2]) + cx)
    v = int((fy * point[1] / point[2]) + cy)
    
    if 0 <= u < width and 0 <= v < height:
        return (u, v)
    return None

def ply_to_video(ply_folder, output_video, fps=30):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Headless rendering

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Get list of .ply files dynamically
    file_list = [os.path.join(ply_folder, f) for f in os.listdir(ply_folder) if f.endswith(".ply") and "filtered_" in f]
    file_list.sort(key=natural_sort_key)  # Sort by numerical index

    if not file_list:
        print("No .ply files found. Exiting.")
        return

    # Load first frame to determine video size
    first_pcd = o3d.io.read_point_cloud(file_list[0])
    pcd.points = first_pcd.points
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)

    img = np.asarray(vis.capture_screen_float_buffer(True)) * 255
    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Define camera intrinsics (assuming a pinhole camera model)
    fx, fy = width / 2, height / 2  # Approximate focal lengths
    cx, cy = width / 2, height / 2  # Principal point (center of image)

    # Process and render each .ply file
    for file in tqdm(file_list):
        new_pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(new_pcd.points)

        if len(points) > 0:
            # Find closest point (smallest Euclidean norm)
            closest_point = points[np.argmin(np.linalg.norm(points, axis=1))]
            projected_point = project_point_to_image(closest_point, (fx, fy, cx, cy), width, height)
        else:
            projected_point = None  # No valid points

        pcd.points = new_pcd.points
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(True)) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

        # Overlay closest point
        if projected_point:
            cv2.circle(img, projected_point, 5, (0, 0, 255), -1)  # Red dot for closest point

        video_writer.write(img)

    video_writer.release()
    vis.destroy_window()  # Ensure window is properly closed
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    ply_folder = "/home/mkhanum/datapipe/Output/vid4_pc"
    output_video = "/home/mkhanum/datapipe/Output/videos/ply_video.mp4"
    fps = 30
    ply_to_video(ply_folder, output_video, fps)
