import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/inbreast-dataset-bi-rads-classification")

print("Path to dataset files:", path)
