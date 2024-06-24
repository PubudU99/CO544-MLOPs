import dagshub
dagshub.init(repo_owner='PubudU99', repo_name='CO544-MLOPs', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)