from src.download import main as download_data
from src.preprocess import main as preprocess_data
from src.train import main as train_models
from src.evaluate import main as evaluate_models


def run_full_pipeline():
    print("Starting MLOps Pipeline...")
    
    print("\n1. Downloading data...")
    download_data()
    
    print("\n2. Preprocessing data...")
    preprocess_data()
    
    print("\n3. Training models...")
    train_models()
    
    print("\n4. Evaluating models...")
    evaluate_models()
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    run_full_pipeline()