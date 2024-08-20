from sutils.image_utils import detectPose
import os ,pickle ,torch 
import argparse, cv2

model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")

def load(model:str ="yolo"):
    """Load the specified model.

    Args:
        model (str, optional): The name of the model to load. Defaults to "yolo".

    Returns:
        The loaded model. If 'yolo' is specified, it loads a YOLO model using Torch Hub.
        Otherwise, it attempts to load a model from a pickle file in the "models" directory.
    """

    if model == "yolo":
        loaded_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\DELL\Desktop\sr\models\best.pt')
        print(f"Loaded yolo model")
        return loaded_model
    else: 
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.pkl'):  
                model_name = file_name.split('.')[0]  
                print(model_name)
                if model == model_name:
                    file_path = os.path.join(model_dir, file_name)
                    with open(file_path, 'rb') as model_file:
                        loaded_model = pickle.load(model_file)
                    print(f"Loaded model: {model_name} from {file_path}")
                    return loaded_model


def pred(img, model = "yolo"):
    """Perform prediction on the given image using the specified model.

    Args:
        img (ndarray): The input image on which prediction is to be made.
        model (str, optional): The name of the model to use for prediction. Defaults to "yolo".

    Returns:
        str: The classification result. For YOLO, it returns the detected object class. 
             For other models, it returns "engaged" or "not engaged" based on the prediction.
    """

    print(model)
    classififcation = " "
    working_model =load(model)
    if model== "yolo":
        print(model ,img.shape)
        results = working_model(img)
        classififcation = results.pandas().xyxy[0][['name']].values.tolist()[0][0]
    else:
        pose_row = detectPose(img,batch=False)
        y_pred = working_model.predict([pose_row])
        classififcation =  "engaged" if y_pred==1 else "not engaged"
    
    print(classififcation)
    return classififcation

def main(image_path: str, model: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred(img, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model prediction on an image.")
    parser.add_argument('--image_path', type =str, default='img_test.png' ,help="Path to the input image.")
    parser.add_argument('--model', type=str, default='yolo', choices=['yolo', 'knn', 'svm', 'rf'],
                    help="Model type to use for prediction. Choose from 'yolo', 'knn', 'svm', or 'rf'. Default is 'yolo'.")
    args = parser.parse_args()
    main(args.image_path, args.model)
