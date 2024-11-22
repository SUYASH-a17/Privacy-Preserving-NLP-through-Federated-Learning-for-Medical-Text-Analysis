import os
from datetime import datetime
import streamlit as st
from scripts.model_deployment import ModelDeployment

# Set Streamlit page configuration
st.set_page_config(
    page_title="Text Classification Model",
    page_icon="📄",
    layout="wide"
)

def main():
    # Set the page title
    st.title("📄 Text Classification Model Deployment")
    st.markdown("""
    Welcome to the Text Classification Model! This application is designed to classify input text into predefined categories.
    """)

    # Path to configuration file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/config.yaml'))

    # Initialize the model deployment class
    deployment = None
    model_status = "Not Loaded"
    try:
        deployment = ModelDeployment(config_path)
        model_status = "Loaded Successfully"
    except Exception as e:
        model_status = f"Failed to Load: {str(e)}"

    # Create tabs for navigation
    tabs = st.tabs(["🏠 Home", "📊 About the Model", "🔧 Health Check", "🔍 Predict"])

    # Tab 1: Home
    with tabs[0]:
        st.markdown("""
        ### Welcome!
        This application uses a machine learning model to classify input text into categories. It's built using PyTorch and Streamlit.
        
        #### Features:
        - **Real-Time Predictions:** Input text and get instant predictions.
        - **Robust Model Architecture:** A neural network trained for accurate classification.
        - **Interactive Interface:** Navigate easily using tabs.
        
        Use the tabs above to explore more about the model or make predictions.
        """)

    # Tab 2: About the Model
    with tabs[1]:
        st.markdown("## About the Model")
        st.markdown("""
        This text classification model categorizes medical and non-medical text into predefined categories.

        ### Model Features:
        - **Architecture:** A feedforward neural network with three layers.
        - **Preprocessing:** Uses vectorization and label encoding.
        - **Fallback Mechanism:** Includes heuristic-based classification for specific keywords.
        
        ### Technologies:
        - **Libraries:** PyTorch, scikit-learn, and Streamlit.
        - **Training:** Custom dataset trained for text classification tasks.
        """)

    # Tab 3: Health Check
    with tabs[2]:
        st.markdown("## Health Check")
        st.markdown("Check the model's status and running environment.")
        st.markdown("---")
        if deployment:
            st.success(f"Model Status: {model_status}")
            st.info(f"Device: {deployment.device}")
            st.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("Model could not be loaded. Please check the logs.")

    # Tab 4: Predict
    with tabs[3]:
        st.markdown("## Predict Text Category")
        st.markdown("Enter text below and click **Predict** to classify it.")
        st.markdown("---")

        # Input text area
        text = st.text_area("Enter your text:", "", key="classification_text_area")

        # Predict button
        if st.button("Predict", key="predict_button"):
            if text and deployment:
                try:
                    prediction_result = deployment.predict(text)
                    st.markdown("### Prediction Result:")
                    st.json(prediction_result)  # Display prediction as JSON
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
            else:
                st.warning("Please enter text and ensure the model is loaded.")

    # Footer
    st.markdown("---")
    st.markdown("Developed by: **Red Coder !!**")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
