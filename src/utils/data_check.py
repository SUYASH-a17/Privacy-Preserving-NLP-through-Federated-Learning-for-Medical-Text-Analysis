import os
import pandas as pd
from typing import Tuple

def check_data_availability() -> Tuple[bool, str]:
    """Check if dataset is available"""
    raw_data_path = 'data/raw'
    
    # Check if directory exists
    if not os.path.exists(raw_data_path):
        return False, "Data directory not found. Please create 'data/raw' directory."
    
    # Check if directory is empty (except .gitkeep)
    files = [f for f in os.listdir(raw_data_path) if f != '.gitkeep']
    if not files:
        return False, ("No dataset found. Please download the dataset and place it in "
                      "'data/raw' directory. Run 'python scripts/download_data.py' "
                      "for instructions.")
    
    return True, "Dataset found."

def validate_dataset(file_path: str) -> Tuple[bool, str]:
    """Validate dataset format"""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['text']  # Add other required columns
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        return True, "Dataset format is valid."
        
    except Exception as e:
        return False, f"Error validating dataset: {str(e)}"
