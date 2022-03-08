## create command using conda env file [an alternative to pip install -r requirements.txt]

```bash
conda env create --prefix ./env -f conda.yaml
```

```bash 
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 -p 1234
```