import mlflow

def calculate_nth_power(x, n):
    return x**n

if __name__ == '__main__':
    with mlflow.start_run():
        x, n = 4, 8
        y = calculate_nth_power(x, n)
        mlflow.log_param("x", x)
        mlflow.log_param("n", n)
        mlflow.log_metric("y", y)