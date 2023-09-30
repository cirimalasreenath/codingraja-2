# Step 1: Data Collection (Assuming you have a dataset of social media posts with labeled sentiments)

# Step 2: Text Preprocessing
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

# Example usage
sample_text = "I love this product! It's amazing!"
cleaned_text = preprocess_text(sample_text)
print(cleaned_text)

# Step 3: Feature Extraction (Using TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have a list of preprocessed texts called 'processed_texts' and corresponding labels in 'labels'
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(processed_texts)

# Step 4: Model Selection (Using SVM)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Assuming you have a list of labels called 'labels'
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 5: Model Training (Already done in the previous step)

# Step 6: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Step 7: Deployment (Optional)
# You can use web frameworks like Flask or Django to create a simple web interface for users to input text for sentiment analysis.
