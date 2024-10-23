import os
import shutil
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .base import LabelstudioConverter
from .utils import *

"""
To make the convert() method flexible and allow it to select different conversion functions
use a strategy pattern. This involves defining a mapping of conversion strategies 
and selecting the appropriate one based on a parameter. 
"""
class LabelstudioConverterPose(LabelstudioConverter):
    def __init__(self, json_path: str, output_dir: str, format_type: str = 'yolo'):
        """
        Initialize with JSON data exported from Label Studio.
        
        :param json_path: Path to the JSON file.
        :param output_dir: Directory to save images and labels.
        """
        super().__init__(json_path)
        self.output_dir = output_dir
        self.format_type = format_type
        self.conversion_strategies = {
            'yolo': self.ls_json_keypoints_to_yolo,
            # Add more formats here as needed -> ex. 'coco': self.ls_json_keypoints_to_coco, etc.
        }

    def convert(self, verbose: bool = False):
        if self.format_type in self.conversion_strategies:
            self.conversion_strategies[self.format_type](verbose)
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}")
        
    def ls_json_keypoints_to_yolo(self, verbose: bool = False):
        """
        Convert the JSON data from Label Studio to YOLO format and save images and labels.
        """
        images_dir = os.path.join(self.output_dir, 'images')
        labels_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for item in tqdm(self.json_data):
            try:
                # get necessary information label_dict
                label_dict = get_keypoints_data(item)
                image_path = convert_labelstudio_image_path(label_dict['image'])

                # Copy image to the images directory
                image_filename = os.path.basename(image_path)
                shutil.copy(image_path, os.path.join(images_dir, image_filename))

                # Prepare YOLO label data
                yolo_labels = ls_keypoints_to_yolo(label_dict)

                # Save YOLO label data to a .txt file
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_filename)
                self.save_yolo_format(yolo_labels, label_path)
            except KeyError as e:
                logging.error("Missing key in data: %s", e)
                if verbose:
                    logging.info("Image_path: %s", image_path)
                    logging.info("Data: %s", label_dict)
            except Exception as e:
                logging.error("Error processing item: %s", e)

    def save_yolo_format(self, data: List[List[float]], file_name: str):
        """
        Save data in YOLO format to a text file.
        
        :param data: The data to be saved, where each sub-list represents a line.
        :param file_name: The name of the file to save the data to.
        """
        with open(file_name, 'w') as file:
            for sublist in data:
                line = ' '.join(map(str, sublist))
                file.write(line + '\n')



def get_keypoints_data(ls_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and organize annotation data from Label Studio JSON into a custom format.
    
    :param ls_data: The Label Studio JSON data containing annotations.
    :return: A dictionary with image information, ID, and lists of bounding boxes and keypoints.
    """
    annotation = ls_data['annotations'][0]
    completed_by = annotation['completed_by']
    updated_by = annotation['updated_by']
    result = annotation['result']
    
    label_dict = {
        'image': ls_data['data']['image'],
        'id': ls_data['id'],
        'keypoints_bbox': [],
        'keypoints': [],
        'completed_by': completed_by,
        'updated_by': updated_by
    }

    for label in result:
        if "rectanglelabels" in label['value']:
            label_dict['keypoints_bbox'].append(prepare_bbox_format(label))
        elif "keypointlabels" in label['value']:
            label_dict['keypoints'].append(prepare_keypoint_format(label))
    return label_dict


def ls_keypoints_to_yolo(keypoints_bbox: List[Dict[str, Any]], 
                        keypoints: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Convert keypoints_bbox, keypoints from label_dict to label in YOLO format.

    :param keypoints_bbox: List of bounding box dictionaries with 'x', 'y', 'width', 'height', 'original_width', 'original_height', and 'id'.
    :param keypoints: List of keypoint dictionaries with 'x', 'y', 'keypointlabels', and 'parentID'.
    :return: List of YOLO formatted entries for each bounding box and its keypoints.
    """
    
    yolo_pose_data = []
    class_index = 0  # Assuming "person" is class 0
    
    # Define the order of keypoints
    keypoint_order = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Create a map from keypoint label to its index in the order
    keypoint_map = {label: index for index, label in enumerate(keypoint_order)}

    # Group keypoints by their parentID (bbox id)
    keypoints_by_bbox = {}
    for kp in keypoints:
        parent_id = kp['parentID']
        if parent_id not in keypoints_by_bbox:
            keypoints_by_bbox[parent_id] = []
        keypoints_by_bbox[parent_id].append(kp)

    # Process each bounding box
    for bbox in keypoints_bbox:
        original_width = bbox['original_width']
        original_height = bbox['original_height']
        
        # Normalize bounding box coordinates from json full
        # / 100 cause it comes from scale 0-100
        x_center = (bbox['x'] + bbox['width'] / 2) / 100
        y_center = (bbox['y'] + bbox['height'] / 2) / 100
        width = bbox['width'] / 100
        height = bbox['height'] / 100
        
        # Start the YOLO pose entry with class index and bbox data
        yolo_entry = [class_index, x_center, y_center, width, height]
        
        # Initialize keypoints list with zeros
        keypoints_list = [[0, 0, 0] for _ in range(17)]

        # Add keypoints for this bbox
        bbox_id = bbox['id']
        if bbox_id in keypoints_by_bbox:
            for kp in keypoints_by_bbox[bbox_id]:
                kp_label = kp['keypointlabels'][0]
                if kp_label in keypoint_map:
                    kp_index = keypoint_map[kp_label]
                    kp_x = kp['x'] / 100
                    kp_y = kp['y'] / 100
                    visibility = 2.0  # Assuming all keypoints are visible
                    keypoints_list[kp_index] = [kp_x, kp_y, visibility]
                    
        # Flatten the keypoints list and append to the yolo entry
        for kp in keypoints_list:
            yolo_entry.extend(kp)
        
        yolo_pose_data.append(yolo_entry)

    return yolo_pose_data