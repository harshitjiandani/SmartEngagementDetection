import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse, os, pickle
from sutils import setup_args
from sutils.image_utils import load_images ,detectPose
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report



def read_pcoordinsates():
    df = load_images()
    df.fillna(0, inplace=True)
    df.head()
    df.tail()
    X = df.drop('class', axis=1) # features
    y = df['class'] # target value
    y  = y.map({"eng":1 , "noteng":0})
    return X,y


def train():
    X, y  = read_pcoordinsates()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=82536)

    pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)) 
    }

    fit_models = {}
    model_dir = os.path.join(os.path.abspath(__file__), "models")
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
