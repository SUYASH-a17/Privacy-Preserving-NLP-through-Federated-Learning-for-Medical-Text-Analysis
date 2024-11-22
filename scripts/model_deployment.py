import os
import logging
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

class SimpleTextClassifier(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleTextClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ModelDeployment:
    """
    Class to handle model loading and prediction.
    """
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['deployment']['device'] if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.load_model_artifacts()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'deployment_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        self.logger = logging.getLogger(f'deployment_{timestamp}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.info(f"Deployment logging initialized - Log file: {log_file}")
        self.logger.info(f"Using device: {self.device}")
        
    def load_model_artifacts(self):
        try:
            self.logger.info("Loading model artifacts...")
            
            # Load vectorizer
            vectorizer_path = os.path.join(self.config['models']['dir'], self.config['models']['vectorizer_file'])
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            input_dim = len(self.vectorizer.get_feature_names_out())
            
            # Load label encoder
            encoder_path = os.path.join(self.config['models']['dir'], self.config['models']['label_encoder_file'])
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            output_dim = len(self.label_encoder.classes_)
            
            # Initialize and load model
            self.model = SimpleTextClassifier(input_dim, 256, 128, output_dim).to(self.device)
            model_path = os.path.join(self.config['models']['dir'], self.config['models']['model_file'])
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.logger.info("Model artifacts loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {str(e)}")
            raise e
    
    def is_medical_text(self, text):
        """
        Check if the input text is related to medical topics.
        """
        medical_keywords = set([
            "vaccine", "vaccination", "treatment", "diagnosis", "symptoms", "therapy",
            "patient", "doctor", "hospital", "disease", "health", "infection",
            "antiviral", "mask", "quarantine", "transmission", "epidemiology",
            "mental health", "nutrition", "chronic", "pediatric", "geriatric",
            "reproductive health", "immunization", "immunize", "immunotherapy",
            "antibiotic", "pharmaceutical", "chemotherapy", "radiation", "remedy",
            "cure", "intervention", "monoclonal antibody", "dialysis", "surgery",
            "transplant", "handwashing", "sanitizer", "hygiene", "prophylactic",
            "screening", "public health", "containment", "disinfectant", "sterilization",
            "airborne", "contagion", "infectious", "vector", "droplet", "contact tracing",
            "super-spreader", "pathogen", "zoonotic", "pandemic", "diagnostics", "pcr",
            "test", "detection", "imaging", "x-ray", "mri", "ct scan", "antigen test",
            "antibody test", "biopsy", "laboratory", "blood test", "ultrasound",
            "epidemic", "case fatality rate", "prevalence", "incidence", "cluster",
            "surveillance", "risk factor", "mortality rate", "disease modeling",
            "depression", "anxiety", "stress", "ptsd", "psychotherapy", "counseling",
            "well-being", "psychiatric", "psychological", "addiction", "substance abuse",
            "behavioral health", "trauma", "bipolar disorder", "diet", "calories",
            "protein", "vitamins", "supplements", "malnutrition", "food security",
            "healthy eating", "dietary guidelines", "fiber", "hydration", "obesity",
            "weight management", "diabetes", "hypertension", "asthma", "copd",
            "arthritis", "cancer", "heart disease", "cardiovascular", "stroke",
            "kidney disease", "autoimmune", "parkinson's", "alzheimer's", "dementia",
            "osteoporosis", "endometriosis", "children's health", "infant", "newborn",
            "adolescent", "growth monitoring", "neonatal", "autism", "birth defects",
            "pediatric oncology", "elderly", "aging", "senior health", "gerontology",
            "nursing home", "fall prevention", "palliative care", "end-of-life care",
            "longevity", "fertility", "ivf", "pregnancy", "maternal health",
            "childbirth", "contraception", "abortion", "sexual health", "menstrual",
            "pcos", "std", "sti", "prenatal care", "postpartum"
            # Add any additional medical keywords here
        ])
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)
    
    def heuristic_classification(self, text):
        """
        Perform heuristic classification based on expanded medical keywords across multiple categories.
        """
        text_lower = text.lower()
        
        # Check if the text is non-medical
        if not self.is_medical_text(text):
            return "Non-Medical", 1.0

        # Vaccines
        if any(keyword in text_lower for keyword in [
            "vaccine", "vaccination", "immunization", "immunize", "inoculation",
            "booster", "dose", "jab", "shot", "immunotherapy"
        ]):
            return "Vaccines", 0.95

        # Treatment
        elif any(keyword in text_lower for keyword in [
            "antiviral", "therapy", "medication", "treatment", "drug", "pharmaceutical",
            "chemotherapy", "radiation", "antibiotic", "remedy", "cure", "intervention",
            "monoclonal antibody", "stem cell transplant", "dialysis", "surgery", "transplant"
        ]):
            return "Treatment", 0.90

        # Prevention
        elif any(keyword in text_lower for keyword in [
            "mask", "quarantine", "prevention", "social distancing", "isolation",
            "handwashing", "sanitizer", "hygiene", "prophylactic", "screening",
            "public health measure", "containment", "disinfectant", "cleaning", "sterilization"
        ]):
            return "Prevention", 0.88

        # Transmission
        elif any(keyword in text_lower for keyword in [
            "transmission", "spread", "airborne", "contagion", "infectious",
            "vector", "droplet", "contact tracing", "super-spreader", "reproduction number",
            "incubation period", "asymptomatic", "pathogen", "zoonotic", "pandemic transmission"
        ]):
            return "Transmission", 0.90

        # Diagnostics
        elif any(keyword in text_lower for keyword in [
            "diagnostics", "pcr", "test", "cr tests", "detection", "diagnosing",
            "screening", "imaging", "x-ray", "mri", "ct scan", "antigen test",
            "antibody test", "biopsy", "laboratory", "blood test", "culture test",
            "ultrasound", "electrocardiogram", "ekg", "blood pressure", "monitoring"
        ]):
            return "Diagnostics", 0.92

        # Epidemiology
        elif any(keyword in text_lower for keyword in [
            "epidemiology", "infection rates", "study", "outbreak", "pandemic",
            "endemic", "epidemic", "case fatality rate", "prevalence", "incidence",
            "cluster", "surveillance", "risk factor", "mortality rate", "disease modeling",
            "public health surveillance"
        ]):
            return "Epidemiology", 0.87

        # Mental Health
        elif any(keyword in text_lower for keyword in [
            "mental health", "depression", "anxiety", "stress", "ptsd", "psychotherapy",
            "counseling", "well-being", "psychiatric", "psychological", "addiction",
            "substance abuse", "behavioral health", "mental illness", "trauma", "bipolar disorder"
        ]):
            return "Mental Health", 0.89

        # Nutrition
        elif any(keyword in text_lower for keyword in [
            "nutrition", "diet", "calories", "protein", "vitamins", "supplements",
            "malnutrition", "food security", "healthy eating", "dietary guidelines",
            "fiber", "hydration", "obesity", "weight management", "nutritional deficiency"
        ]):
            return "Nutrition", 0.88

        # Chronic Diseases
        elif any(keyword in text_lower for keyword in [
            "chronic", "long-term condition", "diabetes", "hypertension", "asthma",
            "copd", "arthritis", "cancer", "heart disease", "cardiovascular",
            "stroke", "kidney disease", "autoimmune", "chronic fatigue", "parkinson's",
            "alzheimer's", "dementia", "osteoporosis", "endometriosis"
        ]):
            return "Chronic Diseases", 0.90

        # Pediatrics
        elif any(keyword in text_lower for keyword in [
            "pediatric", "children's health", "child health", "infant", "newborn",
            "adolescent", "immunization schedule", "growth monitoring", "neonatal",
            "developmental disorders", "autism", "birth defects", "child nutrition",
            "vaccines for children", "pediatric oncology"
        ]):
            return "Pediatrics", 0.88

        # Geriatrics
        elif any(keyword in text_lower for keyword in [
            "geriatric", "elderly", "aging", "senior health", "gerontology",
            "nursing home", "dementia care", "fall prevention", "alzheimer's care",
            "age-related diseases", "palliative care", "end-of-life care", "longevity"
        ]):
            return "Geriatrics", 0.88

        # Reproductive Health
        elif any(keyword in text_lower for keyword in [
            "reproductive health", "fertility", "ivf", "pregnancy", "maternal health",
            "childbirth", "contraception", "abortion", "sexual health", "menstrual",
            "pcos", "std", "sti", "prenatal care", "postpartum"
        ]):
            return "Reproductive Health", 0.90

        # Others
        elif any(keyword in text_lower for keyword in [
            "socioeconomic", "impact", "healthcare disparity", "economic", "policy",
            "access to care", "healthcare system", "inequality", "quality of life",
            "education", "employment", "social determinants", "legislation", "insurance",
            "cost", "ethics", "global health", "public opinion", "technology in health",
            "telemedicine", "remote monitoring", "healthcare innovation"
        ]):
            return "Others", 0.85

        return None, None

    def predict(self, text):
        """
        Make a prediction for the provided text.
        """
        try:
            self.logger.info(f"Processing text: {text[:100]}...")

            # Heuristic classification
            heuristic_label, heuristic_confidence = self.heuristic_classification(text)
            if heuristic_label:
                self.logger.info(f"Heuristic classification used: {heuristic_label} (confidence: {heuristic_confidence})")
                return {
                    "prediction": heuristic_label,
                    "confidence": round(heuristic_confidence, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            # Vectorize input text
            features = self.vectorizer.transform([text])
            feature_array = features.toarray()
            feature_tensor = torch.FloatTensor(feature_array).to(self.device)

            # Check for empty feature vector
            if np.sum(feature_array) == 0:
                self.logger.warning("Empty feature vector generated.")
                return {
                    "prediction": "Uncertain - Input Not Recognizable",
                    "confidence": 0.0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            # Model prediction
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)

            confidence = confidence.item()
            predicted_label = self.label_encoder.inverse_transform([prediction.item()])[0]

            # Confidence threshold
            confidence_threshold = 0.85
            if confidence < confidence_threshold:
                self.logger.info(f"Low confidence prediction.")
                return {
                    "prediction": "Uncertain - Low Confidence",
                    "confidence": round(confidence, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                self.logger.info(f"Prediction made: {predicted_label} (confidence: {confidence:.4f})")
                return {
                    "prediction": predicted_label,
                    "confidence": round(confidence, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
