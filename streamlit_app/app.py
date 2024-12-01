import sys
import os
from datetime import datetime
import streamlit as st

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the ModelDeployment class
try:
    from scripts.model_deployment import ModelDeployment
except ModuleNotFoundError as e:
    st.error(f"Failed to import ModelDeployment: {e}")
    raise

# Set Streamlit page configuration
st.set_page_config(
    page_title="Privacy-Preserving NLP through Federated Learning for Medical Text Analysis",
    page_icon="📄",
    layout="wide"
)

def main():

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
    tabs = st.tabs(["🏠 Home", "📊 About the Model", "📚 About Categories", "🔧 Models Health Check", "🔍 Predict"])


    # Tab 1: Home
    with tabs[0]:
        # Hero Section
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: #2c3e50; font-size: 36px;">📄 Privacy-Preserving NLP through Federated Learning for Medical Text Analysis</h1>
            <p style="color: #6c757d; font-size: 18px;">
                Welcome to the Medical Text Analysis Model! Leveraging AI and Federated Learning for secure and efficient medical text classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Dropdown 1: Overview
        with st.expander("📜 Overview", expanded=True):
            st.markdown("""
            #### Project Overview
            This project demonstrates how natural language processing (NLP) and federated learning can address privacy concerns in medical text analysis.

            **Key Features**:
            - **Privacy-Preserving Learning:** Ensures sensitive medical data remains private using federated learning.
            - **Differential Privacy:** Protects against privacy attacks by adding controlled noise to data and model updates.
            - **Medical Text Analysis:** Performs multi-class classification on medical research abstracts.
            """)

        # Dropdown 2: Motivation
        with st.expander("🎯 Motivation", expanded=False):
            st.markdown("""
            #### Background and Motivation
            - The increasing availability of healthcare data provides opportunities for enhanced patient care and research.
            - Strict regulations like HIPAA and GDPR make centralized data aggregation impractical in the medical domain.
            - Federated learning offers a solution by training models across decentralized datasets without sharing raw data.
            """)

        # Dropdown 3: Objectives
        with st.expander("🚀 Objectives", expanded=False):
            st.markdown("""
            #### Objectives
            - Develop a federated learning model for medical text classification.
            - Incorporate differential privacy mechanisms to safeguard data.
            - Evaluate the federated model's performance against traditional centralized training.
            """)

        # Dropdown 4: Methods
        with st.expander("🔬 Methods", expanded=False):
            st.markdown("""
            #### Methods
            **Model Architecture**:
            - Uses an LSTM-based neural network for sequential text processing.
            - Enhanced with pre-trained embeddings like GloVe or Word2Vec for better semantic understanding.

            **Federated Learning Framework**:
            - Employs the Flower framework for flexible and scalable client simulation.
            - Implements FedAvg for model aggregation across clients.

            **Differential Privacy Integration**:
            - Adds Gaussian noise and applies L2 norm clipping to ensure privacy compliance.
            """)

        # Dropdown 5: Results and Future Work
        with st.expander("📈 Results and Future Work", expanded=False):
            st.markdown("""
            #### Results and Future Work
            **Key Results**:
            - Demonstrated the feasibility of federated learning for medical text analysis with a slight trade-off in accuracy.

            **Future Directions**:
            - Explore transformer-based architectures like BERT for improved contextual understanding.
            - Enhance privacy techniques with adaptive noise addition and secure aggregation protocols.
            """)

    # Tab 2 
    with tabs[1]:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #2c3e50; font-size: 28px;">📊 About the Model</h2>
            <p style="color: #6c757d; font-size: 18px;">
                Learn more about the model architecture, training process, and technologies used for privacy-preserving medical text analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Collapsible Section: Key Features
        with st.expander("📌 Key Features", expanded=True):
            st.markdown("""
            <ul style="font-size: 16px;">
                <li><b>Architecture:</b> A feedforward neural network with three layers.</li>
                <li><b>Preprocessing:</b> Vectorization and label encoding techniques applied.</li>
                <li><b>Privacy Integration:</b> Differential privacy mechanisms safeguard sensitive data.</li>
            </ul>
            """, unsafe_allow_html=True)

        # Collapsible Section: Training Process
        with st.expander("🔬 Training Process", expanded=False):
            st.markdown("""
            - **Training Dataset:** Curated dataset of medical and non-medical texts.
            - **Batch Size:** 32 with early stopping to avoid overfitting.
            - **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score.
            """)

        # Collapsible Section: Privacy Techniques
        with st.expander("🛡️ Privacy-Preserving Techniques", expanded=False):
            st.markdown("""
            - **Federated Learning:** Trains models on decentralized data without sharing raw data between nodes.
            - **Differential Privacy:** Adds Gaussian noise and applies L2 norm clipping to ensure compliance.
            - **Data Anonymization:** Sensitive identifiers are removed or masked before processing.
            """)

        # Collapsible Section: Model Technologies
        with st.expander("🔧 Model Technologies", expanded=False):
            st.markdown("""
            - **Libraries:** PyTorch, Scikit-learn, Streamlit.
            - **Frameworks:** Flower for federated learning.
            - **Embeddings:** Pre-trained embeddings like GloVe or Word2Vec.
            """)

        # Collapsible Section: Future Enhancements
        with st.expander("🚀 Future Enhancements", expanded=False):
            st.markdown("""
            - Explore transformer-based architectures like BERT.
            - Implement real-time prediction capabilities.
            - Enhance privacy techniques with adaptive noise addition.
            """)



        #tab 3 About Categories
        with tabs[2]:
            # Header Section
            st.markdown("""
            <div style="background-color: #f1f7fc; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #2c3e50; font-size: 28px;">📚 About Categories</h2>
                <p style="color: #6c757d; font-size: 18px;">
                    Explore the categories the model predicts and understand their significance in medical text analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Introduction to Categories
            st.markdown("""
            <div style="padding-top: 20px;">
                <p style="font-size: 16px; text-align: justify;">
                    The model classifies medical and non-medical texts into predefined categories based on their content. These categories help in organizing medical research, improving accessibility, and deriving actionable insights.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Detailed Categories with Icons
            st.markdown("<h3>Categories and their Descriptions</h3>", unsafe_allow_html=True)
            categories = [
                {
                    "name": "Vaccines",
                    "icon": "💉",
                    "description": "Covers texts related to vaccination campaigns, immunization, or vaccines.",
                    "example": "E.g., 'The new vaccine rollout has been highly effective.'",
                },
                {
                    "name": "Treatment",
                    "icon": "💊",
                    "description": "Mentions treatments, therapies, or medications for various conditions.",
                    "example": "E.g., 'The patient responded well to the new chemotherapy drug.'",
                },
                {
                    "name": "Prevention",
                    "icon": "🛡️",
                    "description": "Texts about preventive measures like social distancing, mask usage, or hygiene.",
                    "example": "E.g., 'Washing hands regularly helps prevent the spread of infections.'",
                },
                {
                    "name": "Diagnostics",
                    "icon": "🔬",
                    "description": "Focuses on diagnostic methods like imaging, lab tests, and screenings.",
                    "example": "E.g., 'MRI scans were used to detect early signs of the disease.'",
                },
                {
                    "name": "Epidemiology",
                    "icon": "📊",
                    "description": "Covers studies on infection rates, disease spread, and public health trends.",
                    "example": "E.g., 'The epidemiological data shows a steady decline in new cases.'",
                },
                {
                    "name": "Chronic Conditions",
                    "icon": "🩺",
                    "description": "Related to long-term diseases or health management.",
                    "example": "E.g., 'Diabetes management requires regular blood sugar monitoring.'",
                },
                {
                    "name": "Others",
                    "icon": "📁",
                    "description": "General texts that do not fit into the above categories.",
                    "example": "E.g., 'Medical conferences play a vital role in knowledge sharing.'",
                },
            ]

            for category in categories:
                with st.expander(f"{category['icon']} {category['name']}"):
                    st.markdown(f"""
                    - **Description:** {category['description']}
                    - **Example:** {category['example']}
                    """)


    # Tab 4: Health Check
    with tabs[3]:
        st.markdown("## Health Check")
        st.markdown("Check the model's status and running environment.")
        st.markdown("---")
        if deployment:
            st.success(f"Model Status: {model_status}")
            st.info(f"Device: {deployment.device}")
            st.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("Model could not be loaded. Please check the logs.")

    # Tab 5: Predict
    with tabs[4]:
        st.markdown("## Predict Text Category")
        st.markdown("Enter text below and click **Predict** to classify it.")
        st.markdown("---")

        # Input text area
        text = st.text_area("Enter your text:", "", key="classification_text_area")

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
    st.markdown(
        """
        <hr>
        <div style="text-align: center;">
            <p>Developed by: <b>Red Coder !!</b></p>
            <p>Last updated: {}</p>
        </div>
        <hr>
        """.format(datetime.now().strftime('%Y-%m-%d')),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
