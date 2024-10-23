import sys
sys.path.append('/root/pose_estimation_workspace/experiment_workspace')

import os
import random
import yaml
from data.labelstudio_converter.pose import LabelstudioConverterPose     


def main():
    params = yaml.safe_load(open("params.yaml"))["data"]
    converter = LabelstudioConverterPose(json_path=params["annotation_path"], 
                                            output_dir=params["output_path"],
                                            format_type="yolo")
    converter.convert(verbose=False)


if __name__ == '__main__':
    main()