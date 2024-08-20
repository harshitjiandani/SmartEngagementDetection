import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse, os, pickle
from sutils.image_utils import load_images ,detectPose
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report



def read_pcoordinsates(pose_file:bool=False, pose_csv:str=""):
    """Read pose coordinates from images, preprocess the data, and return features and target values.

    Returns:
        X: DataFrame containing the features extracted from pose coordinates.
        y: Series containing the target values mapped to binary labels.
    """
    if pose_file:
        df = pd.read_csv(pose_csv)

    elif pose_file==False:
        df = load_images()

    df.fillna(0, inplace=True)
    X = df.drop(df.columns[0], axis=1) # features
    y = df.iloc[:, 0] # target value
    y  = y.map({"eng":1 , "noteng":0})
    return X,y


def train():
    """Train multiple classifiers on pose coordinate data and save the models.

    Reads pose coordinates from images, splits the data into training and testing sets,
    trains Logistic Regression, Ridge Classifier, Random Forest, Gradient Boosting, and K-Nearest Neighbors classifiers,
    saves the trained models to files, and prints accuracy scores, confusion matrices, and classification reports for each model.
    """
    X, y  = read_pcoordinsates(pose_file=True , pose_csv=r"C:\Users\DELL\Desktop\sr\coordsfinal1.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82536)

    pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)) 
    }

    fit_models = {}
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

        #save
        model_filename = os.path.join(model_dir, f"{algo}_model.pkl")
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Saved {algo} model to {model_filename}")
    
    cms = []
    for a , p in fit_models.items():
        ans = p.predict(X_test)
        print(a,accuracy_score(y_test,ans))
        cms.append(confusion_matrix(y_test, ans, labels=[0,1]))
        print(classification_report(y_test, ans))

if __name__ == "__main__":
    train()