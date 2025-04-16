import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import time

# Check for NLTK resources and download if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Set page config
st.set_page_config(
    page_title="Medical Text Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Define paths
MODEL_PATHS = {
    'svm_model': [
        'svm_model.joblib',
        os.path.join('models', 'classifiers', 'svm_model.joblib'),
        r'D:\Project\Health Care\svm_model.joblib',
        r'D:\Project\Health Care\models\classifiers\svm_model.joblib'
    ],
    'random_forest_model': [
        'random_forest_model.joblib',
        os.path.join('models', 'classifiers', 'random_forest_model.joblib'),
        r'D:\Project\Health Care\random_forest_model.joblib',
        r'D:\Project\Health Care\models\classifiers\random_forest_model.joblib'
    ],
    'tfidf_vectorizer': [
        'tfidf_vectorizer.joblib',
        os.path.join('models', 'classifiers', 'tfidf_vectorizer.joblib'),
        r'D:\Project\Health Care\tfidf_vectorizer.joblib',
        r'D:\Project\Health Care\models\classifiers\tfidf_vectorizer.joblib'
    ]
}

# Function to find and load a model
def load_model(model_name):
    """Find and load a model from possible paths"""
    for path in MODEL_PATHS.get(model_name, []):
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                st.warning(f"Error loading model from {path}: {str(e)}")
    
    return None, None

# Function to check model files
def check_model_files():
    """Check if model files exist and are valid"""
    status = {}
    
    # Check SVM model
    svm_model, svm_path = load_model('svm_model')
    status['SVM'] = "✅ Loaded from: " + svm_path if svm_model else "❌ Not found or invalid"
    
    # Check Random Forest model
    rf_model, rf_path = load_model('random_forest_model')
    status['Random Forest'] = "✅ Loaded from: " + rf_path if rf_model else "❌ Not found or invalid"
    
    # Check TF-IDF vectorizer
    vectorizer, vec_path = load_model('tfidf_vectorizer')
    if vectorizer:
        if hasattr(vectorizer, 'vocabulary_'):
            status['TF-IDF Vectorizer'] = f"✅ Loaded from: {vec_path} (Fitted with {len(vectorizer.vocabulary_)} features)"
        else:
            status['TF-IDF Vectorizer'] = f"⚠️ Loaded from: {vec_path} but not fitted"
    else:
        status['TF-IDF Vectorizer'] = "❌ Not found or invalid"
    
    return status

# Function to convert numeric category to descriptive stay duration label
def get_stay_duration_label(category_num):
    """Convert numeric category to descriptive stay duration label"""
    category_mapping = {
        "0": "Outpatient Care",  # Adding category 0
        "1": "Short Stay",
        "2": "Medium Stay", 
        "3": "Longer Stay"
    }
    # Default to the original value if not in mapping
    return category_mapping.get(str(category_num), f"Category {category_num}")
# Cache preprocessing function for performance
@st.cache_data
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Preprocess text for NLP tasks"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords (if enabled)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Keep negation words and medical terms
        medical_exceptions = {'no', 'not', 'nor', 'pain', 'acute', 'chronic'}
        filtered_stop_words = stop_words - medical_exceptions
        tokens = [token for token in tokens if token not in filtered_stop_words]
    
    # Lemmatize tokens (if enabled)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Standalone summarization function that can be cached
@st.cache_data
def summarize_with_textrank(text, ratio=0.3, min_length=40):
    """Summarize text using TextRank algorithm"""
    if pd.isna(text) or text.strip() == "":
        return ""
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Filter very short sentences
    sentences = [s for s in sentences if len(s) >= min_length]
    
    # If there are too few sentences, return the original text
    if len(sentences) <= 2:
        return text
    
    # Calculate the number of sentences to select based on ratio
    num_sentences = max(2, int(len(sentences) * ratio))
    
    # Create a TF-IDF vectorizer to convert sentences to vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Transform sentences to TF-IDF vectors
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Calculate similarity between all sentence pairs
        similarity_matrix = sentence_vectors * sentence_vectors.T
        
        # Create a graph from the similarity matrix
        nx_graph = nx.from_scipy_sparse_array(similarity_matrix)
        
        # Apply PageRank algorithm
        scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
        
        # Rank sentences based on scores
        ranked_sentences = sorted(((scores[i], i, sentence) for i, sentence in enumerate(sentences)), 
                                 reverse=True)
        
        # Select top sentences
        top_indices = [idx for _, idx, _ in ranked_sentences[:num_sentences]]
        
        # Sort indices to maintain original order
        top_indices.sort()
        
        # Extract the sentences in their original order
        summary_sentences = [sentences[i] for i in top_indices]
        
        # Combine sentences into summary
        summary = ' '.join(summary_sentences)
        
        return summary
        
    except Exception as e:
        # Return a truncated version of original text as fallback
        return ' '.join(sentences[:num_sentences])

# TextRank summarizer implementation
class TextRankSummarizer:
    """Fast TextRank algorithm for extractive summarization"""
    
    def __init__(self):
        self.name = "TextRank"
    
    def summarize(self, text, ratio=0.3, min_length=40):
        # Use the cached helper function instead
        return summarize_with_textrank(text, ratio, min_length)

# Define classification function with improved error handling
@st.cache_data
def classify_medical_text(text, model_name='svm_model'):
    """Classify medical text using the trained models"""
    start_time = time.time()
    
    # Validate model name
    model_key = 'svm_model' if model_name == 'SVM' else 'random_forest_model'
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Try to load models
    try:
        # Load classification model
        model, model_path = load_model(model_key)
        if not model:
            return {"error": f"Model not found for {model_name}"}
        
        # Load vectorizer
        vectorizer, vec_path = load_model('tfidf_vectorizer')
        if not vectorizer:
            return {"error": "TF-IDF vectorizer not found"}
        
        # Check if vectorizer is fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            return {"error": "TF-IDF vectorizer not fitted. Fix required."}
        
        # Transform text to TF-IDF features
        features = vectorizer.transform([processed_text])
        
        # Predict class
        prediction = model.predict(features)[0]
        
        # Get probability estimates if the model supports it
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            class_labels = model.classes_
            
            # Get top 3 classes or all if less than 3
            num_classes = min(3, len(class_labels))
            top_indices = np.argsort(probabilities)[-num_classes:][::-1]
            top_classes = []
            for i in top_indices:
                top_classes.append({
                    'class': class_labels[i],
                    'probability': float(probabilities[i])
                })
        
        # Prepare result
        result = {
            'predicted_class': prediction,
            'model_used': model_name,
            'processing_time': time.time() - start_time
        }
        
        if probabilities is not None:
            result['top_classes'] = top_classes
        
        return result
    
    except Exception as e:
        return {"error": f"Error during classification: {str(e)}"}

# Main app
def main():
    # Set up the title with medical icon
    st.title("🏥 Medical Text Analyzer")
    
    # Check model files and show status
    model_status = check_model_files()
    with st.expander("Model Status", expanded=False):
        for model, status in model_status.items():
            st.text(f"{model}: {status}")
        
        # Add button to run classification.py if models are missing
        missing_models = any("Not found" in status or "not fitted" in status for status in model_status.values())
        if missing_models:
            st.warning("Some models are missing or not properly configured.")
            if st.button("Fix Models"):
                fix_models()
    
    # Main content area with columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Input Medical Text")
        
        # Input options - removed Upload CSV option
        input_option = st.radio(
            "Input Method",
            ["Enter Text", "Use Sample"],
            horizontal=True
        )
        
        text_to_analyze = ""
        
        if input_option == "Enter Text":
            text_to_analyze = st.text_area(
                "Enter medical text",
                height=250,
                placeholder="Enter patient diagnosis, symptoms, treatments, etc..."
            )
        
        elif input_option == "Use Sample":
            samples = {
                "Diabetes": "Patient is a 57-year-old male with history of Type 2 Diabetes Mellitus, hypertension, and hyperlipidemia. Reports polyuria, polydipsia and fatigue for the past month. Recent lab work shows HbA1c of 9.2%. Blood glucose levels consistently elevated at 200-250 mg/dL. Patient is currently on Metformin 1000mg twice daily and Glipizide 10mg daily. Physical exam shows peripheral neuropathy in both feet. Plan to adjust medications and add GLP-1 receptor agonist.",
                "Pneumonia": "Patient is a 42-year-old female presenting with 5-day history of productive cough with yellow-green sputum, fever up to 102°F, and progressive shortness of breath. Oxygen saturation 91% on room air. Chest X-ray shows right lower lobe consolidation consistent with pneumonia. WBC count elevated at 15,500. Started on IV antibiotics with Ceftriaxone and Azithromycin. Patient admitted for IV antibiotics, supplemental oxygen, and respiratory therapy.",
                "Stroke": "72-year-old male with sudden onset right-sided weakness and aphasia that began 2 hours prior to arrival. Patient has history of atrial fibrillation on warfarin, hypertension, and previous TIA 3 years ago. On exam, patient has facial droop, right arm drift, and expressive aphasia. BP 178/95, HR 92 irregular. NIHSS score 8. CT scan shows no acute hemorrhage. Patient given tPA after confirming INR 1.4. Admitted to Neuro ICU for monitoring.",
                "ARDS": "The patient, a 58-year-old female, presents with severe Acute Respiratory Distress Syndrome (ARDS), likely secondary to bacterial sepsis originating from Pseudomonas aeruginosa pneumonia. The onset of symptoms, including rapid breathing, tachypnea, and profound hypoxia, began approximately 48 hours prior to admission. Over the course of this period, the patient has developed acute kidney injury (AKI), respiratory failure, and septic shock, all of which are compounding the severity of her condition. The elevated lactate level of 6.5 mmol/L, coupled with a procalcitonin of 15 ng/mL, indicates a systemic inflammatory response, with multi-organ involvement manifesting as compromised renal function and worsening oxygenation.Plan:ICU Admission:Immediate transfer to the Intensive Care Unit (ICU) is warranted for close monitoring and ventilatory support. The patient will be intubated to facilitate mechanical ventilation, with an initial FiO2 of 100% to optimize oxygenation. Given the severity of the hypoxia and arterial blood gas (ABG) abnormalities (pH 7.28, PaO2 45 mmHg), a careful balance of ventilator settings will be titrated to prevent barotrauma while ensuring adequate oxygen delivery. Sepsis Management:To combat the ongoing infection, broad-spectrum antibiotics will be initiated promptly. Piperacillin-Tazobactam will be started at 4.5 grams every 6 hours, along with Vancomycin at 1 gram every 12 hours. These agents are selected due to their effectiveness against Pseudomonas aeruginosa and other potential pathogens. Cultures, including blood, sputum, and urine, will be obtained immediately to guide further therapy based on sensitivities.Vasopressor Support:In light of the patient’s hypotension (BP 85/58 mmHg), norepinephrine infusion will be initiated to manage shock and maintain a mean arterial pressure (MAP) >65 mmHg. Continuous blood pressure monitoring via an arterial line will help assess the patient’s response to fluid resuscitation and vasoactive agents. The goal is to stabilize circulation and perfuse vital organs, including the kidneys and brain.Renal Support:The patient’s renal function has deteriorated, with a creatinine level of 1.9 mg/dL and a urine output of less than 0.5 mL/kg/hr. Close monitoring of urine output and kidney function will be required. Should the renal function continue to decline, we will initiate Renal Replacement Therapy (RRT) with continuous veno-venous hemodialysis (CVVHD) to support filtration and fluid balance.Fluid Resuscitation:Given the patient’s hypovolemia and hypotension, aggressive intravenous fluid resuscitation with isotonic crystalloids (Normal Saline or Lactated Ringer’s) will be initiated. Fluid boluses of 500 mL every 30 minutes will be administered initially, with careful monitoring of central venous pressure (CVP) and lactate levels to avoid fluid overload, especially given the risk of pulmonary edema in ARDS.Steroid Therapy:To address the underlying inflammation and reduce pulmonary edema, IV corticosteroids (Methylprednisolone) will be started at a dose of 2 mg/kg/day. This will help mitigate the systemic inflammatory response associated with ARDS and reduce the overall duration of mechanical ventilation, as well as decrease the likelihood of progression to more severe forms of lung injury.Nutritional Support:Once the patient’s ventilator settings are stabilized, enteral nutrition will be initiated through a nasogastric tube. This will support metabolic demands and prevent further muscle wasting. The goal is to maintain caloric intake at approximately 25–30 kcal/kg/day, utilizing high-protein formulas to promote tissue repair and immune function during this critical period.Sedation & Analgesia:For patient comfort during intubation and mechanical ventilation, adequate sedation and analgesia will be provided. Fentanyl and propofol will be titrated to achieve the appropriate depth of sedation, ensuring that the patient remains comfortable and cooperative while also preventing self-extubation. Regular assessment of sedation levels will be performed using the Richmond Agitation-Sedation Scale (RASS).Monitor:Continuous monitoring of vital signs, including heart rate, blood pressure, and respiratory parameters, will be conducted. Additionally, urine output will be monitored closely to assess renal perfusion, and lactate levels will be evaluated every 6 hours to track the resolution of shock. Serial arterial blood gas (ABG) analysis will be done to assess oxygenation and acid-base status, and chest X-rays will be repeated daily to monitor the progression of ARDS.Prognosis:The patient’s prognosis is currently guarded, given the severity of her ARDS, septic shock, and multi-organ involvement. Her response to the above therapies will determine whether she can be weaned from mechanical ventilation and whether renal function improves over time. Prolonged ICU care is anticipated, and a multi-disciplinary team, including infectious disease specialists, nephrologists, and respiratory therapists, will be involved in her ongoing management."
            }   
            
            selected_sample = st.selectbox("Select sample text", list(samples.keys()))
            text_to_analyze = samples[selected_sample]
            st.text_area("Sample text", text_to_analyze, height=200, disabled=True)
        
        # Analysis options
        st.subheader("Analysis Options")
        
        # Summary options
        summary_ratio = st.slider(
            "Summary Compression",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Lower values = shorter summary"
        )
        
        # Classification options
        classifier = st.radio(
            "Classification Model",
            ["SVM", "Random Forest"],
            horizontal=True
        )
        
        # Submit button
        analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
        
    with col2:
        if analyze_button and text_to_analyze.strip():
            st.subheader("Analysis Results")
            
            # Start processing with a spinner
            with st.spinner("Analyzing medical text..."):
                # Process text
                
                # 1. Generate summary with TextRank
                start_time = time.time()
                summarizer = TextRankSummarizer()
                summary = summarizer.summarize(text_to_analyze, ratio=summary_ratio)
                summarization_time = time.time() - start_time
                
                # 2. Classify text
                classification_result = classify_medical_text(text_to_analyze, model_name=classifier)
            
            # Display results
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.markdown("### Summary")
                st.info(summary)
                
                # Summary metrics
                original_len = len(text_to_analyze)
                summary_len = len(summary)
                compression = (1 - summary_len / original_len) * 100
                
                st.metric("Compression Rate", f"{compression:.1f}%", f"{summarization_time:.2f} seconds")
                
            with results_col2:
                st.markdown("### Classification")
                
                if "error" in classification_result:
                    st.error(classification_result["error"])
                    
                    # If error is related to models, offer to fix
                    if "not found" in classification_result["error"].lower() or "not fitted" in classification_result["error"].lower():
                        if st.button("Fix Classification Models"):
                            fix_models()
                else:
                    # Display stay duration label instead of numeric category
                    stay_label = get_stay_duration_label(classification_result['predicted_class'])
                    st.success(f"**{stay_label}**")
                    st.metric("Processing Time", f"{classification_result.get('processing_time', 0):.2f} seconds")
                    
                    # Show confidence if available
                    if "top_classes" in classification_result and classification_result['model_used'] == 'SVM':
                        # Create bar chart for confidence
                        fig, ax = plt.subplots(figsize=(7, 4))
                        
                        # Convert class numbers to descriptive labels
                        classes = [get_stay_duration_label(item["class"]) for item in classification_result["top_classes"]]
                        probs = [item["probability"] for item in classification_result["top_classes"]]
                        
                        # Create horizontal bar chart
                        bars = ax.barh(classes, probs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Confidence")
                        ax.set_title("Length of Stay Prediction Confidence")
                        
                        # Add labels
                        for i, v in enumerate(probs):
                            ax.text(v + 0.01, i, f'{v:.2f}', va='center')
                        
                        st.pyplot(fig)
        else:
            # Instructions when no analysis is running
            st.info("Enter medical text and click 'Analyze Text' to see results.")
            
            # Display information about the application
            st.markdown("""
            ### About This Tool
            
            This medical text analyzer helps healthcare professionals to:
            
            - **Summarize** lengthy medical documents into concise summaries
            - **Classify** medical texts by predicted length of stay
            
            ### Available Models
            
            - **TextRank** (Summarization): Fast graph-based algorithm that extracts key sentences
            - **SVM & Random Forest** (Classification): Machine learning models trained on medical texts
            
            ### How to Use
            
            1. Enter medical text or select a sample
            2. Adjust summary compression as needed
            3. Choose a classification model
            4. Click "Analyze Text" to process
            
            ### Stay Duration Categories
            
            - **Short Stay**: Brief hospitalization or outpatient care
            - **Medium Stay**: Moderate length hospitalization
            - **Longer Stay**: Extended hospitalization for complex cases
            """)
    
    # Footer
    st.divider()
    st.caption("Medical Text Analyzer v1.0 | Optimized for speed and accuracy")

# Function to fix models by running classification.py
def fix_models():
    """Function to run the classification script to fix models"""
    try:
        st.info("Attempting to fix models by running classification script...")
        
        # Define classification script content
        classification_script = """
import pandas as pd
import numpy as np
import re
import os
import nltk
import joblib
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Check for NLTK resources and download if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Function to preprocess text
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    if pd.isna(text) or text == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        medical_exceptions = {'no', 'not', 'nor', 'pain', 'acute', 'chronic'}
        filtered_stop_words = stop_words - medical_exceptions
        tokens = [token for token in tokens if token not in filtered_stop_words]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Main function
def main():
    # Define paths
    data_path = r'D:\\Project\\Health Care\\dataset\\mimic3d.csv'
    models_dir = r'D:\\Project\\Health Care\\models\\classifiers'
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    print("Loading dataset...")
    
    try:
        # Try to load the dataset
        df = pd.read_csv(data_path)
        
        print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # In case the dataset cannot be loaded or doesn't exist, create sample data
        if df.shape[0] == 0:
            raise ValueError("Dataset is empty, creating sample data instead")
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}. Creating sample data instead.")
        
        # Create sample data
        data = {
            'AdmitDiagnosis': [
                "Diabetes mellitus type 2 with hyperglycemia",
                "Acute myocardial infarction",
                "Community acquired pneumonia",
                "Congestive heart failure",
                "Chronic obstructive pulmonary disease",
                "Urinary tract infection",
                "Acute appendicitis",
                "Cerebrovascular accident",
                "Gastroenteritis",
                "Asthma exacerbation"
            ],
            'LOSgroupNum': [1, 2, 1, 3, 2, 1, 1, 3, 1, 2]
        }
        df = pd.DataFrame(data)
    
    # Set columns
    text_column = 'AdmitDiagnosis'
    target_column = 'LOSgroupNum'
    
    if text_column not in df.columns:
        # Find a suitable text column
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 20:
                text_column = col
                break
    
    if target_column not in df.columns:
        # Find a suitable target column
        for col in df.columns:
            if col != text_column and df[col].dtype != 'object' and df[col].nunique() <= 10:
                target_column = col
                break
    
    print(f"Using text column: {text_column}")
    print(f"Using target column: {target_column}")
    
    # Clean the data
    df[text_column] = df[text_column].fillna('').astype(str)
    df[target_column] = df[target_column].fillna(-1)
    
    # Convert target to string for classification
    if pd.api.types.is_numeric_dtype(df[target_column]):
        df[target_column] = df[target_column].astype(int).astype(str)
    
    # Keep only rows with non-empty text
    df = df[df[text_column].str.strip() != '']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    df['encoded_target'] = label_encoder.fit_transform(df[target_column])
    
    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['encoded_target'],
        test_size=0.2,
        random_state=42
    )
    
    # Create and fit TF-IDF vectorizer
    print("Creating and fitting TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
    
    # Train SVM model
    print("Training SVM model...")
    svm_model = LinearSVC(C=1.0, max_iter=10000)
    svm_model.fit(X_train_vec, y_train)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_vec, y_train)
    
    # Save models
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(svm_model, os.path.join(models_dir, 'svm_model.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.joblib'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    
    # Copy to root dir
    root_dir = r'D:\\Project\\Health Care'
    joblib.dump(tfidf_vectorizer, os.path.join(root_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(svm_model, os.path.join(root_dir, 'svm_model.joblib'))
    joblib.dump(rf_model, os.path.join(root_dir, 'random_forest_model.joblib'))
    
    print("Models created and saved successfully!")

if __name__ == "__main__":
    main()
"""
        
        # Save the script to a temporary file
        script_path = "temp_classification.py"
        with open(script_path, "w") as f:
            f.write(classification_script)
        
        # Run the script
        import subprocess
        process = subprocess.Popen(["python", script_path], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Show output in real-time
        output_placeholder = st.empty()
        output = ""
        
        for line in process.stdout:
            output += line
            output_placeholder.text(output)
        
        # Wait for process to complete
        process.wait()
        
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        
        # Check if successful
        if process.returncode == 0:
            st.success("Models successfully fixed! Please refresh the app.")
            st.experimental_rerun()
        else:
            st.error("Error fixing models. See output for details.")
    
    except Exception as e:
        st.error(f"Error fixing models: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
