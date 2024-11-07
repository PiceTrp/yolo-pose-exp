import sys
sys.path.append('/root/pose_estimation_workspace/experiment_workspace')

import os
import shutil
import random
from pathlib import Path
import yaml
from ultralytics import YOLO

# Visibiity adjustment function
from data.adjust_visibility import get_correct_visibility_label
from data.yolo_dataset import YoloDataset


def split_data(image_dir, label_dir, split_ratio=0.8, seed=42):
    """
    Splits the dataset into training and validation sets by moving files.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing labels.
        split_ratio (float): Ratio of the dataset to be used for training.

    Creates:
        train/images, train/labels, val/images, val/labels directories with respective data.
    """
    random.seed(seed)
    # Ensure the directories exist
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    assert image_dir.exists() and label_dir.exists(), "Image or label directory does not exist."

    # Get list of image files
    image_files = list(image_dir.glob('*.jpg'))  # Assuming images are in .jpg format
    random.shuffle(image_files)

    # Calculate split index
    split_index = int(len(image_files) * split_ratio)

    # Create train and val directories
    for split in ['train', 'val']:
        (image_dir.parent / split / 'images').mkdir(parents=True, exist_ok=True)
        (label_dir.parent / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Split and copy files
    for i, image_file in enumerate(image_files):
        split = 'train' if i < split_index else 'val'
        label_file = label_dir / (image_file.stem + '.txt')
        
        if label_file.exists():
            shutil.copy(str(image_file), str(image_dir.parent / split / 'images' / image_file.name))
            shutil.copy(str(label_file), str(label_dir.parent / split / 'labels' / label_file.name))

    # Remove original directories
    # shutil.rmtree(image_dir)
    # shutil.rmtree(label_dir)


def create_data_yaml(data_path):
    """
    Creates a data.yaml file for YOLO dataset configuration using a dictionary.

    Args:
        data_path (str): Path to the dataset root directory.
    """
    data = {
        'path': data_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': '',
        'kpt_shape': [17, 3],
        'flip_idx': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    }
    names_section = yaml.dump({'names': {0: 'person'}}, default_flow_style=False)

    yaml_path = Path(data_path) / 'data.yaml'
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=None)
        file.write(names_section)


def main():
    # Parse arguments
    params = yaml.safe_load(open("params.yaml"))
    seed = params['prepare']['seed']

    # setup data
    current_dir = os.getcwd()
    dataset_name = params['data']['output_path']
    data_path = os.path.join(current_dir, dataset_name) # Path to the dataset root directory
    dataset = YoloDataset(current_dir, dataset_name)

    # Adjust visibility
    model_path = os.path.join(current_dir, 'assets', 'yolo11x-pose.pt')
    model = YOLO(model_path)
    save_label_dir = os.path.join(data_path, 'labels')
    for i in range(len(dataset.image_paths)):
        try:
            # save txt label with visibility adjusted
            label_result = get_correct_visibility_label(image_path=dataset.image_paths[i], 
                                                        label_path=dataset.label_paths[i], 
                                                        model=model, 
                                                        save_txt_dir=save_label_dir, 
                                                        save=True, verbose=False)
        except:
            # meaning no label file
            print("No label File... Save empty label")
            with open(os.path.join(save_label_dir, os.path.basename(dataset.label_paths[i])), 'w') as file:
                pass

    # Split the dataset into train and validation sets
    image_dir = os.path.join(data_path, 'images')
    label_dir = os.path.join(data_path, 'labels')
    split_data(image_dir, label_dir, params['prepare']['split'], seed)

    # Create a data.yaml file for YOLO dataset configuration
    create_data_yaml(data_path)


if __name__ == '__main__':
    main()