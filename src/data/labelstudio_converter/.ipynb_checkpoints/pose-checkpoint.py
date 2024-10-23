from .base import LabelstudioConverter


class LabelstudioConverterPose(LabelstudioConverter):
    def __init__(self, json_path, output_dir):
        """
        Initialize with JSON data exported from Label Studio.
        
        :param json_path: Path to the JSON file.
        :param output_dir: Directory to save images and labels.
        """
        self.json_data = self.load_data(json_path)
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def load_data(self, file_path):
        """
        Load and parse the JSON data.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def convert(self):
        """
        Convert the data to YOLO format and save images and labels.
        """
        for item in self.json_data:
            image_path = item['image']
            keypoints_bbox = item['keypoints_bbox']
            keypoints = item['keypoints']

            # Copy image to the images directory
            image_filename = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(self.images_dir, image_filename))

            # Prepare YOLO label data
            yolo_labels = self.convert_to_yolo_format(keypoints_bbox, keypoints)

            # Save YOLO label data to a .txt file
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_filename)
            with open(label_path, 'w') as label_file:
                label_file.write(yolo_labels)

    def convert_to_yolo_format(self, keypoints_bbox, keypoints):
        """
        Convert keypoints and bounding boxes to YOLO format.
        """
        yolo_data = []
        for bbox in keypoints_bbox:
            cls = 0  # Assuming 'person' class is 0
            x_center = bbox['x'] + bbox['width'] / 2
            y_center = bbox['y'] + bbox['height'] / 2
            width = bbox['width']
            height = bbox['height']

            # Normalize coordinates
            x_center /= bbox['original_width']
            y_center /= bbox['original_height']
            width /= bbox['original_width']
            height /= bbox['original_height']

            # Prepare keypoints data
            keypoints_data = []
            for kp in keypoints:
                kp_x = kp['x'] / kp['original_width']
                kp_y = kp['y'] / kp['original_height']
                visible = 1  # Assuming all keypoints are visible
                keypoints_data.extend([kp_x, kp_y, visible])

            # Combine all data into a single line
            yolo_line = [cls, x_center, y_center, width, height] + keypoints_data
            yolo_data.append(' '.join(map(str, yolo_line)))

        return '\n'.join(yolo_data)