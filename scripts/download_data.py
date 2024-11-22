import os
import requests
import argparse
from tqdm import tqdm

def download_dataset(output_dir: str = 'data/raw'):
    """
    Download the CORD-19 dataset
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Please download the CORD-19 dataset from:")
    print("https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge")
    print(f"\nPlace the downloaded files in: {output_dir}")
    print("\nAlternatively, you can use your own medical text dataset in CSV format with columns:")
    print("- text: The medical text content")
    print("- [other columns as needed]")

def create_data_readme():
    """Create README files for data directories"""
    data_readme = """# Data Directory

## Structure
- `raw/`: Contains the raw dataset files
- `processed/`: Contains preprocessed dataset files

## Dataset
This project uses the CORD-19 dataset. Due to size limitations, the dataset is not included in this repository.

### How to get the dataset:
1. Download the CORD-19 dataset from Kaggle:
   https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge

2. Place the downloaded files in the `raw` directory.

### Alternative Dataset
You can also use your own medical text dataset in CSV format with the following columns:
- text: The medical text content
- [other columns as needed]
"""

    # Write README files
    with open('data/README.md', 'w') as f:
        f.write(data_readme)

def main():
    parser = argparse.ArgumentParser(description='Download and setup dataset')
    parser.add_argument('--output-dir', default='data/raw', 
                      help='Directory to store the dataset')
    
    args = parser.parse_args()
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Create README
    create_data_readme()
    
    # Download dataset
    download_dataset(args.output_dir)

if __name__ == "__main__":
    main()