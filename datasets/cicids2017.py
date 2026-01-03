import kagglehub

# Download latest version
path = kagglehub.dataset_download("ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")

print("Path to dataset files:", path)