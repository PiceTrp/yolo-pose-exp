import os
import mlflow
from ultralytics import YOLO


def setup_mlflow(config, verbose=True):
    # Set MLflow tracking URI and token
    # mlflow.set_tracking_uri("https://mlflow.connectedtech.co.th")
    # mlflow.set_experiment("YOLOv11-pose")
    os.environ["MLFLOW_TRACKING_URI"] = config['mlflow']['MLFLOW_TRACKING_URI']

    # Set MinIO credentials in the Python script
    os.environ["AWS_ACCESS_KEY_ID"] = config['mlflow']['AWS_ACCESS_KEY_ID']
    os.environ["AWS_SECRET_ACCESS_KEY"] = config['mlflow']['AWS_SECRET_ACCESS_KEY']
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config['mlflow']['MLFLOW_S3_ENDPOINT_URL']
    
    # Verify that the values are set correctly
    if verbose:
        print("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI"))
        print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
        print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
        print("MLFLOW_S3_ENDPOINT_URL:", os.getenv("MLFLOW_S3_ENDPOINT_URL"))


def main():
    params_data = yaml.safe_load(open("params.yaml"))["data"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]

    # setup - mlflow
    setup_mlflow(params_train)
    # setup - model YOLO
    model = YOLO("./yolo11x-pose.pt")
    # train
    results = model.train(data=os.path.join(params_data['output_path'], "data.yaml"), 
                          epochs=params_train['yolo']['epochs'], 
                          imgsz=params_train['yolo']['img_size'],
                          batch_size=params_train['yolo']['batch_size'],
                          project=params_train['yolo']['project'],
                          name=params_train['yolo']['name'])


if __name__ == "__main__":
    main()