import cv2
import os


def images_to_mp4(folder_path, output_video_path, fps):
    # Get all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Sort files if needed (e.g., by name)
    image_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Check if there are any images
    if not image_files:
        print("No images found in the specified folder.")
        exit()

    # Read the first image to get dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video_path}")
