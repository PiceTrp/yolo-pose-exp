import os
import random
import yaml

from data.LabelstudioConverterPose import LabelstudioConverterPose


def main():
    params = yaml.safe_load(open("params.yaml"))["data"]
    converter = LabelstudioConverterPose(json_path=params.annotation_path, output_dir=params.output_path)
    converter.convert(format_type="yolo", verbose=True)


if __name__ == '__main__':
    main()