import logging     #shift tab
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow 
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)          #???


#print(sys.argv)

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2 

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(34)

    # fetch data
    csv_url=('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
    try:
        data=pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception('no file exists')

    # ML
    train, test=train_test_split(data)
    train_y,test_y=train[['quality']], test[['quality']]
    train_x,test_x=train.drop(['quality'],axis=1), test.drop(['quality'],axis=1)

    alpha= float(sys.argv[1]) if len(sys.argv)>1 else .5         # sys   ???
    l1_ratio= float(sys.argv[2]) if len(sys.argv)>2 else .5

    with mlflow.start_run():
        regression=ElasticNet(alpha=alpha, l1_ratio= l1_ratio, random_state=34)
        regression.fit(train_x, train_y)
        pre= regression.predict(test_x)
        (rmse,mae,r2)=eval_metrics(test_y,pre)

        print(f'elasticnet model(alpha={alpha:f}, l1={l1_ratio:f})')  #
        print('rmse:%s'%rmse)
        print('mae:%s'%mae)
        print('r2:%s'%r2)


        #mlflow
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae',mae)
        mlflow.log_metric('r2',r2)

        predictions=regression.predict(train_x)  
        signature=infer_signature(train_x, predictions)   #output is a dataframe like table; why training set?
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        print('tracking uri:', mlflow.get_tracking_uri())
        print('artifact uri:', mlflow.get_artifact_uri()) 

        if tracking_url_type_store !='file':
            mlflow.sklearn.log_model(
                regression,'model', registered_model_name='elasticnet', signature=signature
            )
        else:
            mlflow.sklearn.log_model(regression,'model',signature=signature)

 


