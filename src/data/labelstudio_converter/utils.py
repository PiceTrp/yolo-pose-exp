def prepare_bbox_format(bbox):
    """
    convert original bbox value from json-full to my custom json value
    still no preprocess in this step
    """
    bbox_dict = {}
    bbox_dict['x'] = bbox['value']['x']
    bbox_dict['y'] = bbox['value']['y']
    bbox_dict['width'] = bbox['value']['width']
    bbox_dict['height'] = bbox['value']['height']
    bbox_dict['rotation'] = bbox['image_rotation']
    bbox_dict['rectanglelabels'] = bbox['value']['rectanglelabels']
    bbox_dict['original_width'] = bbox['original_width']
    bbox_dict['original_height'] = bbox['original_height']
    # add-on box id
    bbox_dict['id'] = bbox['id']
    bbox_dict['type'] = bbox['type'] # rectanglelabels
    return bbox_dict


def prepare_keypoint_format(keypoint):
    """
    convert original keypoint value from json-full to my custom json value
    still no preprocess in this step
    """
    keypoint_dict = {}
    keypoint_dict['x'] = keypoint['value']['x']
    keypoint_dict['y'] = keypoint['value']['y']
    keypoint_dict['width'] = keypoint['value']['width']
    keypoint_dict['keypointlabels'] = keypoint['value']['keypointlabels']
    keypoint_dict['original_width'] = keypoint['original_width']
    # add-on parent ids
    keypoint_dict['parentID'] = keypoint['parentID']
    keypoint_dict['type'] = keypoint['type'] # keypointlabels
    return keypoint_dict


def convert_labelstudio_image_path(path):
    """
    Convert image path from Label Studio JSON to a correct absolute path.
    
    :param path: The original path from Label Studio JSON.
    :return: The transformed absolute path.
    :raises FileNotFoundError: If the transformed path does not exist.
    """
    # Define the prefix to replace
    prefix_to_replace = "/data/local-files/?d="
    
    # Get the home directory from the environment variable
    home_dir = os.environ.get("HOME")
    
    if not home_dir:
        raise EnvironmentError("HOME environment variable is not set.")
    
    # Replace the prefix with the home directory path
    if path.startswith(prefix_to_replace):
        relative_path = path[len(prefix_to_replace):]
        absolute_path = os.path.join(home_dir, relative_path)
        
        # Check if the path exists
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"The path does not exist: {absolute_path}")
            
        return absolute_path
    else:
        raise ValueError(f"Path does not start with expected prefix: {prefix_to_replace}")


def lsxywh(bbox):
    """
    Convert bounding box coordinates from a value between 0-100  to pixel values.

    Parameters:
    bbox (dict): A dictionary containing the bounding box data with keys 'x', 'y', 'width', 'height',
                 'original_width', and 'original_height'.

    Returns:
    dict: A dictionary with the converted pixel values for 'x', 'y', 'width', and 'height'.
    """
    original_width = bbox['original_width']
    original_height = bbox['original_height']

    # Convert each value from the 0-100 scale to pixel values
    x_pixel = (bbox['x'] / 100) * original_width
    y_pixel = (bbox['y'] / 100) * original_height
    width_pixel = (bbox['width'] / 100) * original_width
    height_pixel = (bbox['height'] / 100) * original_height

    return {
        'x': x_pixel,
        'y': y_pixel,
        'width': width_pixel,
        'height': height_pixel
    }