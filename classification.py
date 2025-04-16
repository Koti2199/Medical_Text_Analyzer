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

def main():
    # Define paths
    data_path = r'D:\Project\Health Care\dataset\mimic3d.csv'
    models_dir = r'D:\Project\Health Care\models\classifiers'
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    print(f"Loading dataset from {data_path}...")
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Based on the sample data, we can identify:
        # Text column: 'AdmitDiagnosis'
        # Potential target columns: 'admit_type' or 'LOSgroupNum' or 'admit_location'
        
        text_column = 'AdmitDiagnosis'
        target_column = 'LOSgroupNum'  # Can be changed to 'admit_type' or other categorical variable
        
        print(f"Using text column: {text_column}")
        print(f"Using target column: {target_column}")
        
        # Clean the data
        print("Cleaning and preprocessing data...")
        
        # For text column, replace 'na' with empty string and handle NaN values
        df[text_column] = df[text_column].replace('na', '')
        df[text_column] = df[text_column].fillna('')
        
        # Add AdmitProcedure to text if available and not 'na'
        if 'AdmitProcedure' in df.columns:
            df['combined_text'] = df[text_column]
            mask = (df['AdmitProcedure'].notna()) & (df['AdmitProcedure'] != 'na')
            df.loc[mask, 'combined_text'] = df.loc[mask, text_column] + ' ' + df.loc[mask, 'AdmitProcedure']
            text_column = 'combined_text'
        
        # Handle target column - ensure it's a categorical variable
        df[target_column] = df[target_column].fillna(-1)  # Fill missing values with -1 or another placeholder
        
        # Convert target to string if it's numeric for classification
        if pd.api.types.is_numeric_dtype(df[target_column]):
            df[target_column] = df[target_column].astype(int).astype(str)
        
        # Keep only rows with non-empty text
        df = df[df[text_column].str.strip() != '']
        print(f"Dataset has {df.shape[0]} non-empty rows after cleaning")
        
        # Print class distribution
        print("Class distribution:")
        class_counts = df[target_column].value_counts()
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} samples")
        
        # Encode target labels
        print("Encoding target labels...")
        label_encoder = LabelEncoder()
        df['encoded_target'] = label_encoder.fit_transform(df[target_column])
        
        # Preprocess text
        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(preprocess_text)
        
        # Split data into train and test sets
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['encoded_target'],
            test_size=0.2,
            random_state=42,
            stratify=df['encoded_target']
        )
        
        # Create and fit TF-IDF vectorizer
        print("Creating and fitting TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
        X_train_vec = tfidf_vectorizer.fit_transform(X_train)
        X_test_vec = tfidf_vectorizer.transform(X_test)
        
        # Check if vectorizer was properly fitted
        if not hasattr(tfidf_vectorizer, 'vocabulary_'):
            raise ValueError("TF-IDF vectorizer was not properly fitted!")
        
        print(f"TF-IDF vectorizer fitted successfully with {len(tfidf_vectorizer.vocabulary_)} features")
        
        # Save TF-IDF vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        print(f"Saving TF-IDF vectorizer to {vectorizer_path}...")
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        
        # Save label encoder
        encoder_path = os.path.join(models_dir, 'label_encoder.joblib')
        print(f"Saving label encoder to {encoder_path}...")
        joblib.dump(label_encoder, encoder_path)
        
        # Train SVM model
        print("Training SVM model...")
        svm_model = LinearSVC(C=1.0, max_iter=10000)
        svm_model.fit(X_train_vec, y_train)
        
        # Evaluate SVM model
        y_pred_svm = svm_model.predict(X_test_vec)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        print("SVM Classification Report:")
        print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
        
        # Save SVM model
        svm_path = os.path.join(models_dir, 'svm_model.joblib')
        print(f"Saving SVM model to {svm_path}...")
        joblib.dump(svm_model, svm_path)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_vec, y_train)
        
        # Evaluate Random Forest model
        y_pred_rf = rf_model.predict(X_test_vec)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print("Random Forest Classification Report:")
        print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
        
        # Save Random Forest model
        rf_path = os.path.join(models_dir, 'random_forest_model.joblib')
        print(f"Saving Random Forest model to {rf_path}...")
        joblib.dump(rf_model, rf_path)
        
        # Create additional copies of the files at the project root for compatibility with Streamlit app
        root_dir = r'D:\Project\Health Care'
        
        # Copy vectorizer to root
        root_vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.joblib')
        print(f"Copying TF-IDF vectorizer to {root_vectorizer_path}...")
        joblib.dump(tfidf_vectorizer, root_vectorizer_path)
        
        # Copy SVM model to root
        root_svm_path = os.path.join(root_dir, 'svm_model.joblib')
        print(f"Copying SVM model to {root_svm_path}...")
        joblib.dump(svm_model, root_svm_path)
        
        # Copy Random Forest model to root
        root_rf_path = os.path.join(root_dir, 'random_forest_model.joblib')
        print(f"Copying Random Forest model to {root_rf_path}...")
        joblib.dump(rf_model, root_rf_path)
        
        # Print summary
        print("\nTraining and evaluation completed successfully!")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print("\nSaved models:")
        print(f"  TF-IDF Vectorizer: {vectorizer_path} (copy at {root_vectorizer_path})")
        print(f"  SVM Model: {svm_path} (copy at {root_svm_path})")
        print(f"  Random Forest Model: {rf_path} (copy at {root_rf_path})")
        print(f"  Label Encoder: {encoder_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()