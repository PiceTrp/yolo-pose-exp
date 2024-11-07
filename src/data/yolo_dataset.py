import numpy as np
import os
from glob import glob
from tqdm.notebook import tqdm


class YoloDataset:
    def __init__(self, base_dir, dataset_name, data=None):
        """
        data_split = "train" or "val" or "test" only
        """
        self.base_dir = base_dir
        if data:
            self.dataset_dir = os.path.join(self.base_dir, dataset_name, data)
        else:
            self.dataset_dir = os.path.join(self.base_dir, dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.labels_dir = os.path.join(self.dataset_dir, "labels")
        self.image_paths = sorted(glob(f"{self.images_dir}/*"))
        self.label_paths = sorted(glob(f"{self.labels_dir}/*"))
        
        if len(self.image_paths) != len(self.label_paths):
            self.create_empty_txt()
        else:
            print("Images are corresponding to Labels")
        
    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return (f"Base Directory:     {self.base_dir}\n"
                f"Dataset Directory:  {self.dataset_dir}\n"
                f"Images Directory:   {self.images_dir}\n"
                f"Labels Directory:   {self.labels_dir}\n"
                f"Total images:       {len(self.image_paths)}\n"
                f"Total labels:       {len(self.label_paths)}\n")
                # f"check_correct_files: {self.check_correct_files()}")

    def check_same_filename(self, image_path, label_path):
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        label_filename = os.path.splitext(os.path.basename(label_path))[0]
        return image_filename == label_filename
    
    def check_correct_files(self):
        check = True
        for i in tqdm(range(len(self.image_paths))):
            if not self.check_same_filename(self.image_paths[i], self.label_paths[i]):
                check = False
        return check

    # In case we have images that have no labels, label files might be missing, we will create empty .txt files
    def filter_missing_labels(self):
        image_filenames = set(map(lambda i: os.path.splitext(os.path.basename(i))[0], self.image_paths))
        label_filenames = set(map(lambda i: os.path.splitext(os.path.basename(i))[0], self.label_paths))
        missing_labels = set(map(lambda i: os.path.join(self.labels_dir, f"{i}.txt"), list(image_filenames - label_filenames)))
        return missing_labels

    def create_empty_txt(self):
        missing_labels = self.filter_missing_labels()
        for label_path in tqdm(missing_labels):
            with open(label_path, 'w') as file:
                pass

