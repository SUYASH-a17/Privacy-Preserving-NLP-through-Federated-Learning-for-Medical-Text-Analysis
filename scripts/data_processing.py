# scripts/data_processing.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import yaml
import os
import pickle

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df, text_column='text', label_column='label', max_features=10000):
    """
    Preprocess the data by vectorizing texts and encoding labels.
    """
    texts = df[text_column].values
    labels = df[label_column].values
    
    # Vectorize texts
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    return X, y, vectorizer, label_encoder

def save_artifacts(vectorizer, label_encoder, config):
    """
    Save preprocessing artifacts.
    """
    models_dir = config['models']['dir']
    os.makedirs(models_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, config['models']['vectorizer_file'])
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    label_encoder_path = os.path.join(models_dir, config['models']['label_encoder_file'])
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def main():
    config = load_config('configs/config.yaml')
    df = load_data(config['data']['raw_path'])
    X, y, vectorizer, label_encoder = preprocess_data(
        df,
        max_features=10000
    )
    save_artifacts(vectorizer, label_encoder, config)
    # Optionally save processed data
    # ...

if __name__ == "__main__":
    main()
