
# Smart Engagement Detection

This project provides insights into the applicability of various machine learning
and deep learning algorithms in user
engagement detection and can assist
telecommunication companies in
developing effective customer retention
strategies.
## Dataset
The dataset initially consisted of 1822
manually labelled images collected from
various sources belonging to two classes
engaged (735 images) and not-engaged
(1087 images).

![image](https://github.com/user-attachments/assets/0c701709-fc5e-44f3-a282-f003912efccf)

To enhance certain images and balance
the dataset, Stable Diffusion Image-to-Image (Img2Img) with ControlNet was
used. This method effectively increased
the number of images in the minority
class and improved the quality of certain
images in the dataset.

![image](https://github.com/user-attachments/assets/4edff9ac-f640-444e-90fa-49e5852c1281)


## Inference
Making predictions on a single image using FastAPI call and fine-tuned YOLOv5 model:
```
python main.py --host <HOST> --port <PORT>
```
Alternatively the model used for the prediction can be chosen (Choose from 'yolo', 'knn', 'svm', or 'rf'. Default is 'yolo'):
```
python predict.py --image_path <IMAGE_PATH> --model <MODEL_TYPE>
```


## Train

You can retrain the models by running
```
python train.py
```

