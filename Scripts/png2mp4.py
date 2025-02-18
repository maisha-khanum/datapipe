import cv2
import os
import argparse

def stitch_images_to_video(input_folder, output_file, fps=30):
    images = []
    for i in range(278):  # Only include frames 0 to 150
        # filename = os.path.join(input_folder, f"seg_frame_color_{i:03d}.png")
        filename = os.path.join(input_folder, f"color_{i:03d}.png")
        if os.path.exists(filename):
            images.append(filename)
        else:
            print(f"Warning: {filename} not found, skipping...")

    if not images:
        print("No valid images found. Exiting.")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Stitch PNG images into an MP4 video.")
    # parser.add_argument("input_folder", type=str, help="Path to folder containing PNG images")
    # parser.add_argument("output_file", type=str, help="Output MP4 filename")
    # parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    # args = parser.parse_args()

    # stitch_images_to_video(args.input_folder, args.output_file, args.fps)

    # input_folder = "/home/mkhanum/datapipe/Output/vid4"
    # output_file = "/home/mkhanum/datapipe/Output/videos/seg_mask.mp4"

    input_folder = "/home/mkhanum/smartbelt/aligned_vid4"
    output_file = "/home/mkhanum/smartbelt/videos/vid4.mp4"
    fps = 30
    
    stitch_images_to_video(input_folder, output_file, fps)
