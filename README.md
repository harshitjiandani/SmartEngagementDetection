
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

### Models Trained

The `train.py` script trains multiple models: Logistic Regression, Ridge Classifier, Random Forest, Gradient Boosting, and K-Nearest Neighbors. These models cover a range of classification techniques suitable for different data complexities.

The script allows you to choose between using a pre-existing pose coordinates file or your own image dataset for training. This flexibility is important since the provided dataset is not the full version, enabling you to customize the training process according to your data availability.

```
python train.py --pose_file <POSE_FILE_FLAG> --pose_csv <POSE_CSV> --img_folder <IMG_FOLDER>
```
**Note**: YOLO training is not provided in this script. If you wish to train YOLO for pose detection, you will need to create bounding boxes on your Datset, and fine-tune the model using YOLOv5 , and further refine it. Once done, replace the existing YOLO model in the `models` directory named `best.pt` with your newly trained version. 


