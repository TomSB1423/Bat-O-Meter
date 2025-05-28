import os
from pathlib import Path

from PIL import Image


def create_gif(image_folder, output_path, duration=60, loop=0):
    # Get a list of all image files in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    # Sort images to maintain the order
    images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Load the images
    frames = [Image.open(os.path.join(image_folder, image)).convert("RGB") for image in images]

    print("Opened frames")

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # Duration between frames in milliseconds
        loop=loop,  # 0 means loop forever
    )

    print(f"GIF created successfully and saved to {output_path}")


# Example usage
image_folder = os.path.join(Path(__file__).parent.parent, ".temp")
output_path = "output.gif"  # Replace with your desired output file path
create_gif(image_folder, output_path)
