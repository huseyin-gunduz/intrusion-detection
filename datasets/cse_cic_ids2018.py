import kagglehub

# Download latest version
path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")

print("Path to dataset files:", path)