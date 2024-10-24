from ultralytics.utils.plotting import Annotator
from ultralytics.utils.ops import xywhn2xyxy, scale_coords
from ultralytics.engine.results import Keypoints

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def yolopose2dict(text):
    """
    Convert YOLO pose label.txt data to a list of dictionaries containing object & keypoints keys.
    
    Args:
      text (str): Multiline string containing YOLO pose data.
    
    Returns:
      list: A list of dictionaries, each containing 'object' and 'keypoints' data.
    """
    lines = text.strip().split("\n")
    result = []

    for line in lines:
        data = line.split(" ")
        # object data
        object_data = data[:5]
        if len(object_data) != 5:
            raise ValueError(f"Expected 5 elements for object data, got {len(object_data)}")
        object_label = [int(object_data[i]) if i < 1 else np.float32(object_data[i]) for i in range(len(object_data))] # [class_id, center_x, center_y, width, height]: bbox of object, human class 0
        # keypoints data
        keypoints_data = data[5:] # 2d keypoints in 3-dimensional data: [x, y, visible]
        if len(keypoints_data) != 51:
            raise ValueError(f"Expected 51 elements for keypoints data, got {len(keypoints_data)}")
        keypoints = [[keypoints_data[i], keypoints_data[i+1], keypoints_data[i+2]] for i in range(0, len(keypoints_data), 3)] # pairs of 3 of all keypoint coordinates
        # append
        result.append({"object": object_label, "keypoints": keypoints})
    return result


def denormalize_coordinates(coords, width, height):
    """
    De-normalize the coordinates from normalized values to pixel values.
    
    Args:
      coords (numpy.ndarray): Normalized coordinates of shape (n, 2).
      width (int): Original width of the image.
      height (int): Original height of the image.
    
    Returns:
      numpy.ndarray: De-normalized coordinates of shape (n, 2).
    """
    # Multiply the normalized coordinates by the image dimensions to get pixel values
    coords[:, 0] *= width   # Rescale x-coordinates
    coords[:, 1] *= height  # Rescale y-coordinates
    return coords


def annotate_image(image_path, label_path, box_label="human", line_width=4, 
                   box_color=(0, 0, 255), txt_color=(255, 255, 255), radius=5, conf_thres=0.25, 
                   save=False, fname=None, verbose=False, plot=False):
    """
    Annotate an image with bounding boxes and keypoints from YOLO pose labels.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO pose label file.
        line_width (int, optional): Line width for drawing boxes and keypoints. Default is 4.
        box_color (tuple, optional): Color for the bounding box in (R, G, B). Default is (0, 0, 255).
        txt_color (tuple, optional): Color for the text label in (R, G, B). Default is (255, 255, 255).
        radius (int, optional): Radius for keypoint circles. Default is 5.
        conf_thres (float, optional): Confidence threshold for drawing keypoints. Default is 0.25.
    """
    # Read label data
    with open(label_path, 'r') as file:
        yolo_pose_label = file.read()
    
    label_data = yolopose2dict(yolo_pose_label)

    # Load and prepare the image with PIL.Image
    image = Image.open(image_path)
    w, h = image.size

    # Draw annotation
    annotator = Annotator(image, line_width=line_width, pil=True)
    for item in label_data:
        box = np.array(item['object'][1:], dtype=np.float32)
        box = xywhn2xyxy(box, w, h)
        # Draw bbox
        annotator.box_label(box, label=box_label, color=box_color, txt_color=txt_color, rotated=False)

        # Rescale keypoints coordinate at the scale of the original image
        keypoints = np.array(item['keypoints'], dtype=np.float32)
        coords_2d = keypoints[:, :2]
        rescaled_coords = denormalize_coordinates(coords_2d, w, h)
        keypoints[:, :2] = rescaled_coords
        # Draw keypoints and joint lines
        annotator.kpts(keypoints, shape=(w, h), radius=radius, kpt_line=True, conf_thres=conf_thres)

    # Verbose
    if verbose:
        print("Label Data:", label_data)
        print("Image Shape:", image.shape)
        print("Example Box:", box)
        print("Example Keypoints:", keypoints)
        print("Keypoints Shape:", keypoints.shape)

    if save:
        annotator.im.save(fname) # save 
    # else:
    #     return np.asarray(annotator.im)

    if plot:
        # Show result image
        plt.imshow(annotator.result())
        plt.title(f'{os.path.basename(image_path)}')
        plt.axis('off')  # Hide axes
        plt.show()


if __name__ == '__main__':
    image_path = "/root/pose_estimation_workspace/experiment_workspace/data/exp1/train/images/20240827-084711-5m_01-29_01-48_Code_11-31-42-t-nm_frame_0000.jpg"
    label_path = "/root/pose_estimation_workspace/experiment_workspace/data/exp1/train/labels/20240827-084711-5m_01-29_01-48_Code_11-31-42-t-nm_frame_0000.txt"
    annotate_image(image_path, label_path, box_label="human", line_width=4, 
                   box_color=(0, 0, 255), txt_color=(255, 255, 255), radius=5, conf_thres=0.25, 
                   save=True, 
                   fname="/root/pose_estimation_workspace/experiment_workspace/assets/example_plot.jpg", 
                   verbose=False, plot=False)