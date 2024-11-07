import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from shapely.geometry import box
import numpy as np
import os
from PIL import Image

# Set NumPy print options to display more decimal places
np.set_printoptions(precision=16)

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
        object_label = [int(float(object_data[0])) if i < 1 else np.float32(object_data[i]) for i in range(len(object_data))] # [class_id, center_x, center_y, width, height]: bbox of object, human class 0
        
        # keypoints data
        keypoints_data = data[5:] # 2d keypoints in 3-dimensional data: [x, y, visible]
        if len(keypoints_data) != 51:
            raise ValueError(f"Expected 51 elements for keypoints data, got {len(keypoints_data)}")
        keypoints = [[np.float64(keypoints_data[i]), np.float64(keypoints_data[i+1]), float(keypoints_data[i+2])] for i in range(0, len(keypoints_data), 3)] # pairs of 3 of all keypoint coordinates
        # append
        result.append({"object": object_label, "keypoints": keypoints})
    return result


def convert_yolo_to_xyxy(yolo_results, image_width, image_height):
    """
    Convert YOLO bounding box format to pixel coordinates.

    Parameters:
    - yolo_results: List of YOLO results, each in the format [class, center_x, center_y, width, height]
    - image_width: Original width of the image
    - image_height: Original height of the image

    Returns:
    - List of bounding boxes in the format [class, x_min, y_min, x_max, y_max]
    """
    bboxes = []
    for result in yolo_results:
        class_id, center_x, center_y, width, height = result

        # Convert normalized coordinates to pixel coordinates
        x_min = int((center_x - width / 2) * image_width)
        y_min = int((center_y - height / 2) * image_height)
        x_max = int((center_x + width / 2) * image_width)
        y_max = int((center_y + height / 2) * image_height)

        # Append the converted bounding box to the list
        bboxes.append([class_id, x_min, y_min, x_max, y_max])

    return bboxes


def convert_normalized_to_original_scale(data_list, image_width, image_height):
    """
    Convert normalized keypoint coordinates to original image scale.

    Parameters:
    - data_list: A list of shape [n, 17, 3] where each entry is [x, y, visible_value]
    - image_width: The original width of the image
    - image_height: The original height of the image

    Returns:
    - A numpy array of the same shape with x and y converted to the original scale
    """
    # Convert the list to a numpy array
    data = np.array(data_list, dtype=float)
    
    # Create a copy of the data to avoid modifying the original array
    original_scale_data = np.copy(data)

    # Scale x coordinates by image width
    original_scale_data[:, :, 0] *= image_width

    # Scale y coordinates by image height
    original_scale_data[:, :, 1] *= image_height

    return original_scale_data


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes using shapely.
    
    Parameters:
    - box1: The first bounding box [x_min, y_min, x_max, y_max]
    - box2: The second bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
    - IoU: Intersection over Union of the two boxes
    """
    # Create shapely boxes
    box1_shapely = box(*box1)
    box2_shapely = box(*box2)
    
    # Calculate intersection and union
    intersection_area = box1_shapely.intersection(box2_shapely).area
    union_area = box1_shapely.union(box2_shapely).area
    
    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def find_best_yolo_boxes(annotation_bboxes, yolo_boxes):
    """
    Find the YOLO bounding box with the highest IoU for each annotated bounding box.
    
    Parameters:
    - annotation_bboxes: List of annotated bounding boxes [x_min, y_min, x_max, y_max]
    - yolo_boxes: List of YOLO detection bounding boxes [x_min, y_min, x_max, y_max]
    
    Returns:
    - best_yolo_boxes_dict: Dictionary with index of best YOLO box as key and coordinates as value
    """
    best_yolo_boxes_dict = {}

    for annotation_index, annotation in enumerate(annotation_bboxes):
        best_iou = 0
        best_box_index = None

        for yolo_index, yolo_box in enumerate(yolo_boxes):
            iou = calculate_iou(annotation, yolo_box)
            if iou > best_iou:
                best_iou = iou
                best_box_index = yolo_index

        if best_box_index is not None:
            best_yolo_boxes_dict[annotation_index] = best_box_index
        else:
            best_yolo_boxes_dict[annotation_index] = -1

    return best_yolo_boxes_dict


def normalize_coordinates(coords, width, height):
    """
    Normalize the keypoints coordinates
    
    Args:
      coords (numpy.ndarray): Pixel coordinates of shape (n, 2).
      width (int): Original width of the image.
      height (int): Original height of the image.
    
    Returns:
      numpy.ndarray: De-normalized coordinates of shape (n, 2).
    """
    # Divide the normalized coordinates by the image dimensions to get Normalize values
    coords[:, 0] /= width   # Normalize x-coordinates
    coords[:, 1] /= height  # Normalize y-coordinates
    return coords


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


def get_normalize_keypoints(annot_keypoints, image_width, image_height):
    normalized_annot_keypoints = []
    for annot_keypoint in annot_keypoints:
        data = np.array(annot_keypoint, dtype=np.float64)
        coordinates = data[:, :2]
        normalized_coordinates = normalize_coordinates(coordinates, image_width, image_height)
        data[:, :2] = normalized_coordinates
        data = data.tolist()
        normalized_annot_keypoints.append(data)
    return normalized_annot_keypoints


def save_yolo_format_data(bboxes, keypoints, filename='yolo_format_data.txt'):
    """
    Concatenates bounding boxes and keypoints into a single array and saves it to a text file.

    Parameters:
    - bboxes: List or NumPy array of bounding boxes in YOLO format, shape [n, 5].
    - keypoints: List or NumPy array of keypoints in YOLO format, shape [n, 17, 3].
    - filename: The name of the file to save the data to.
    """
    # Convert lists to NumPy arrays for easier manipulation
    bboxes = np.array(bboxes, dtype=np.float64)
    keypoints = np.array(keypoints, dtype=np.float64)

    # Ensure the keypoints are flattened to match the required shape
    keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)

    # Concatenate bboxes and keypoints
    data = np.concatenate((bboxes, keypoints_flattened), axis=1)

    # Save the data to a text file
    np.savetxt(filename, data, fmt='%.16f')
    print(f"Data saved to {filename}")


def get_correct_visibility_label(image_path, label_path, model, save_txt_dir, save=False, verbose=False):
    image_width, image_height = Image.open(image_path).size
    keypoint_order = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    keypoint_dict = {index: keypoint for index, keypoint in enumerate(keypoint_order)}
    
    # Our annotation
    with open(label_path, 'r') as file:
        data = file.read()
    annotation = yolopose2dict(data)
    annot_bboxes = []
    annot_keypoints = []
    for item in annotation:
        annot_bboxes.append(item['object'])
        annot_keypoints.append(item['keypoints'])

    # preserve normalized data
    original_annot_bboxes = annot_bboxes
        
    # rescale
    annot_bboxes = convert_yolo_to_xyxy(annot_bboxes, image_width, image_height)
    annot_bboxes = [item[1:] for item in annot_bboxes] # exclude class, it's all human class [0]
    annot_keypoints = convert_normalized_to_original_scale(annot_keypoints, image_width, image_height).tolist()
    if verbose:
        print("Our bboxes")
        display(annot_bboxes)
        print("Our Keypoints")
        display(annot_keypoints)
        
    # Yolo Result
    results = model.predict(image_path, conf=0.05, classes=[0], verbose=False)
    detections = results[0].boxes
    yolo_boxes = detections.xyxy.to('cpu').numpy().tolist()
    keypoints = results[0].keypoints
    yolo_skeletons = keypoints.data.to('cpu').numpy().tolist()
    # print("Yolo bboxes")
    # display(yolo_boxes)
    # print("Yolo Keypoints")
    # display(yolo_skeletons)
    
    # Check Same bbox to compare
    matched_bbox = find_best_yolo_boxes(annot_bboxes, yolo_boxes)
    print(matched_bbox)
    
    # Process visibility adjustment
    for annot_index, yolo_matched_index in matched_bbox.items():
        if yolo_matched_index == -1:
            print("No match found... Assume visibility value...")
            annot_skeleton = annot_keypoints[annot_index]
            # Update the third element of each item in knee & ankle
            for keypoint_index in range(13,17):
                if annot_skeleton[keypoint_index][0] == 0.0 and annot_skeleton[keypoint_index][1] == 0.0:
                    # remaining visibility = 0
                    pass
                else:
                    # Change 3rd element in annot_skeleton to visibility = 1.0
                    annot_skeleton[keypoint_index][2] = 1.0
                    
        else:
            annot_skeleton = annot_keypoints[annot_index]
            yolo_skeleton = yolo_skeletons[yolo_matched_index]
        
            # Check if the inner dimensions match
            if len(annot_skeleton) != len(yolo_skeleton):
                print(f"Item {i} has different sizes in the two lists.")
                continue
        
            # Find indices where the third element of each item in yolo_skeleton is less than 0.5.
            to_change_visibility = []
            for j in range(len(yolo_skeleton)):
                # Check if the third element is less than 0.5
                if yolo_skeleton[j][2] < 0.5:
                    to_change_visibility.append(j)
            print(f"Indice to change: {to_change_visibility}")
            print(f"The classes are: {[keypoint_dict[i] for i in to_change_visibility]}")
    
            # Update the third element of each item in annot_skeleton following to_change_visibility
            for keypoint_index in to_change_visibility:
                if annot_skeleton[keypoint_index][0] == 0.0 and annot_skeleton[keypoint_index][1] == 0.0:
                    # remaining visibility = 0
                    pass
                else:
                    # Change 3rd element in annot_skeleton to visibility = 1.0
                    annot_skeleton[keypoint_index][2] = 1.0
            
    # print result
    if verbose:
        print("Our Keypoints - After adjustment")
        display(annot_keypoints)

    # Preprocess back to normalized bboxes and normalized keypoints after adjustment visibility value
    normalized_annot_keypoints = get_normalize_keypoints(annot_keypoints, image_width, image_height)
    if verbose:
        print("normalize bboxes & keypoints")
        print(original_annot_bboxes) # normalized bboxes
        display(normalized_annot_keypoints)

    # save yolo pose txt file after preprocessing visibility value
    if save:
        if not os.path.exists(save_txt_dir):
            os.makedirs(save_txt_dir, exist_ok=True)
        save_txt_path = os.path.join(save_txt_dir, os.path.basename(label_path))
        save_yolo_format_data(original_annot_bboxes, normalized_annot_keypoints, filename=save_txt_path)

    return {"object":original_annot_bboxes, "keypoints":normalized_annot_keypoints}