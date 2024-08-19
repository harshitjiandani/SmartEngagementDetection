import cv2
import os 
import numpy as np
import mediapipe as mp 
import csv
import pandas as pd

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,  static_image_mode = True  , ) 





def detectPose(image, label):
    """
    Perform pose detection on the input image.

    Args:
        image: The input image for pose detection.
        label: The label associated with the image.

    Returns:
        None
    """

    original_image = image.copy()
    
    image_in_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = holistic.process(image_in_RGB)

    if results.pose_landmarks:    
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    else:print("no pose")

    if results.face_landmarks:   
        face = results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
    else:print("no face")

    try:
        allrows= []
        row = pose_row+face_row
        row.insert(0, label)
        
        allrows.append(row)
        with open('posecordinates.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
            
            
    except:
        print("error retrieving pose")
    



def load_images(img_path:str = os.path.join(os.getcwd(), 'images')):
    """
   Load images from the specified directory, perform pose detection, and print results.
    
    Args:
        img_path: Path to the directory containing image folders.
    
    Returns:
        df: A DataFrame containing the pose detection results from all images.
    
    Raises:
        Exception: If an error occurs during file processing.
    """

    label_dict = {"engaged": "eng", "not engaged": "noteng"}
    for folder_name in os.listdir(img_path):
        counter = 0
        label = label_dict.get(folder_name)

        if folder_name != ".DS_Store":
            try:
                for file_name in os.listdir(os.path.join(img_path, folder_name)):
                    if file_name != ".DS_Store":
                        image_path = os.path.join(img_path, folder_name, file_name)
                        output = cv2.imread(image_path)

                        if output is not None:
                            detectPose(output, label)
                            counter += 1
                            print(counter, file_name, label)
                        else:
                            print("Warning: Unable to read", file_name)
    
            except Exception as e:
                print(f"Error processing file: {file_name}. Error: {str(e)}")
        
    df = pd.read_csv('posecordinates.csv')
    return df
               