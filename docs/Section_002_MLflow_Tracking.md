# Section 2: MLflow Tracking

## Simple ML model -

We have implemented a simple ML model to showcase the experiment tracking concept used in MLflow-

### Source code
??? Note "simple_ML_model.py"

    ```python linenums="1"
    import os
    import argparse
    import pandas as pd
    import numpy as np

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet
    from urllib.parse import urlparse
    import mlflow
    import mlflow.sklearn


    def get_data():
        URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

        try:
            df = pd.read_csv(URL, sep=";")
            return df
        except Exception as e:
            raise e

    def evaluate(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def main(alpha, l1_ratio):

        df = get_data()

        train, test = train_test_split(df)

        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)

        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # mlflow 
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            pred = lr.predict(test_x)

            rmse, mae, r2 = evaluate(test_y, pred)

            print(f"Elastic net params: alpha: {alpha}, l1_ratio: {l1_ratio}")
            print(f"Elastic net metric: rmse:{rmse}, mae: {mae}, r2:{r2}")

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)


    if __name__=="__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--alpha", "-a", type=float, default=0.5)
        args.add_argument("--l1_ratio", "-l1", type=float, default=0.5)
        parsed_args = args.parse_args()
        try:
            main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_ratio)
        except Exception as e:
            raise e
    ```

* Source repository - [Click here](https://github.com/c17hawke/mlflow-introduction/tree/main/mlflow-codebase/simple-ML-model)


### Concept of Runs
MLflow Tracking is based on runs.
Runs are executions of some piece of data science code.
A Run can record the following :

*   Code Version 
*   Start & End Time
*   Source
*   Parameters
*   Metrics
*   Artifacts


## Logging our simple ML model using

In this lecture it has been shown that how we can log our model for every execution or experiment-

### Source code
??? Note "simple_ML_model_2.py"

    ```python hl_lines="61" linenums="1"
    import os
    import argparse
    import pandas as pd
    import numpy as np

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet
    from urllib.parse import urlparse
    import mlflow
    import mlflow.sklearn


    def get_data():
        URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

        try:
            df = pd.read_csv(URL, sep=";")
            return df
        except Exception as e:
            raise e

    def evaluate(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def main(alpha, l1_ratio):

        df = get_data()

        train, test = train_test_split(df)

        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)

        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # mlflow 
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            pred = lr.predict(test_x)

            rmse, mae, r2 = evaluate(test_y, pred)

            print(f"Elastic net params: alpha: {alpha}, l1_ratio: {l1_ratio}")
            print(f"Elastic net metric: rmse:{rmse}, mae: {mae}, r2:{r2}")

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # mlflow model logging 
            mlflow.sklearn.log_model(lr, "model")


    if __name__=="__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--alpha", "-a", type=float, default=0.5)
        args.add_argument("--l1_ratio", "-l1", type=float, default=0.5)
        parsed_args = args.parse_args()
        try:
            main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_ratio)
        except Exception as e:
            raise e
    ```

## Exploring UI of MLflow

!!! info
    Refer video lecture for this in oneNeuron platform for UI exploration

??? Note "runs.py"

    ```python
    import numpy as np
    import os

    alpha_s=np.linspace(0.1, 1.0, 5)
    l1_ratios=np.linspace(0.1, 1.0, 5)

    for alpha in alpha_s:
    for l1 in l1_ratios:
        os.system(f"python simple_ML_model_2.py -a {alpha} -l1 {l1}")
    ```