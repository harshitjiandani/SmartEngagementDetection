from sutils.image_utils import detectPose
import os ,pickle

model_dir = os.path.join(os.path.abspath(__file__), "models")
loaded_models = {}

def load(model ="knn"):
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.pkl'):  
            model_name = file_name.split('.')[0]  
            if model == model_name:
                file_path = os.path.join(model_dir, file_name)
                # Load 
                with open(file_path, 'rb') as model_file:
                    loaded_model = pickle.load(model_file)
            
                print(f"Loaded model: {model_name} from {file_path}")
                return loaded_model


def pred(img, model ="knn"):
    pose_row = detectPose(img)
    working_model =load(model)
    y_pred = working_model.predict([pose_row])
    print("engaged" if y_pred==1 else print("not engaged"))
 
    

            
    


