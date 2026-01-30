import numpy as np
from PIL import Image
import imageio
import plotly.io as pio
import os


def fig_to_frame(fig):
    """Convert a matplotlib figure to a numpy array frame.
    
    Renders the figure to an RGB buffer and converts it to a numpy array
    suitable for video creation or image processing.
    
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure to convert.
        
    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with dtype uint8.
    """
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return frame


def plotly_fig_to_frame(fig):
    """Convert a plotly figure to a PIL Image frame.
    
    Saves the figure to a temporary PNG file and reads it back as a PIL Image.
    
    Args:
        fig (plotly.graph_objects.Figure): The plotly figure to convert.
        
    Returns:
        PIL.Image.Image: The figure as a PIL Image.
    """
    pio.write_image(fig, 'tmp.png')  # Save as PNG
    frame = Image.open('tmp.png')
    os.remove('tmp.png')
    return frame


def resize_image_fixed_aspect(image, target_width=None, target_height=None):
    """Resize an image while preserving aspect ratio.
    
    Resizes the image to match either the target width or target height,
    computing the other dimension to maintain the original aspect ratio.
    
    Args:
        image (PIL.Image.Image): The image to resize.
        target_width (int, optional): Desired width. If provided, height is
            computed to maintain aspect ratio. Defaults to None.
        target_height (int, optional): Desired height. If provided, width is
            computed to maintain aspect ratio. Defaults to None.
            
    Returns:
        PIL.Image.Image: The resized image.
        
    Raises:
        ValueError: If neither target_width nor target_height is specified.
    """
    # Get the original dimensions
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # If target_width is provided, compute the corresponding height to maintain the aspect ratio
    if target_width is not None:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    # If target_height is provided, compute the corresponding width to maintain the aspect ratio
    elif target_height is not None:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        raise ValueError("You must specify either target_width or target_height.")

    # Resize the image while preserving the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    return resized_image


def save_frames_as_video(frames, output_path, fps=2):
    """Save a list of frames as a video file.
    
    Args:
        frames (list): List of frames, each either a numpy array or PIL.Image.
        output_path (str): Path to save the output video file.
        fps (int, optional): Frames per second for the video. Defaults to 2.
    """
    frames = [Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame for frame in frames]
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')


def crop_frame(frame, left=0.05, right=0.95, top=0.1, bottom=0.9):
    """Crop frame using normalized coordinates from edges.
    
    Args:
        frame: Input frame/image array
        left: Left edge position in normalized coords (0-1)
        right: Right edge position in normalized coords (0-1) 
        top: Top edge position in normalized coords (0-1)
        bottom: Bottom edge position in normalized coords (0-1)
        
    Returns:
        Cropped frame array
    """
    h, w = frame.shape[:2]
    h_start = int(h * top)
    h_end = int(h * bottom)
    w_start = int(w * left) 
    w_end = int(w * right)
    return frame[h_start:h_end, w_start:w_end]