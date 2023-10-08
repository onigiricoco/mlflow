import mlflow
if __name__== '__main__':
    #mlflow.create_experiment(name='e2')
    #mlflow.set_experiment()
    with mlflow.start_run():
        mlflow.log_param('b',3)
        mlflow.log_metric('a',4)