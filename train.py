import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
  # Enter the path to the yaml file of the model to be trained
  # The example is as follows
  # model.load('yolov8n.pt')  
  results = model.train(
    data='datasets/VisDrone/visDrone.yaml',  
    epochs=300,  
    batch=4,  
    patience=30, 
    imgsz=640,  
    workers=4,  
    device= '0', 
    optimizer='SGD',  
    amp= False, 
    cache=False,  
    name='visDrone-gs_v8s',
    exist_ok=True,
  )