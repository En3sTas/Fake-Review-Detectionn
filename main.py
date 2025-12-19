import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import joblib 

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imbalanced-learn (SMOTE)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)      
    return text

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def calculate_sentiment_scores(x):
    return x.apply(get_sentiment).values.reshape(-1, 1)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("--- 1. LOADING DATA ---")
try:
    df = pd.read_csv("fake reviews dataset.csv", encoding="utf-8")
    print(f"Dataset loaded successfully. Total samples: {len(df)}")
except FileNotFoundError:
    print("ERROR: 'fake reviews dataset.csv' not found.")
    exit()

print("\n--- 2. PRE-PROCESSING ---")
df['text_cleaned'] = df['text_'].apply(clean_text)

le = LabelEncoder()
y = le.fit_transform(df['label'])
classes = list(le.classes_)
print(f"Classes encoded: {classes}")

print("\n--- 3. DATA SPLITTING ---")
X = df['text_cleaned'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# ==========================================
# 3. MODEL TRAINING & COMPARISON
# ==========================================
print("\n" + "="*50)
print("--- 4. TRAINING & COMPARING MODELS ---")
print("="*50)

# Feature Union: TF-IDF + Sentiment Score
combined_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('sentiment', FunctionTransformer(calculate_sentiment_scores, validate=False))
])

# Define Configurations for LR and RF
model_configs = [
    {
        'name': 'Logistic Regression',
        'pipeline': ImbPipeline([
            ('features', combined_features),
            ('smote', SMOTE(random_state=42)),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        'params': {'clf__C': [0.1, 1, 10]}
    },
    {
        'name': 'Random Forest',
        'pipeline': ImbPipeline([
            ('features', combined_features),
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
        ]),
        'params': {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [10, 20, None]
        }
    }
]

best_global_score = 0
best_global_model = None
best_global_name = ""
results_summary = []

for config in model_configs:
    print(f"\n⏳ TRAINING: {config['name']} (Please wait)...")
    
    rs = RandomizedSearchCV(
        config['pipeline'],
        param_distributions=config['params'],
        n_iter=3,
        cv=3,
        scoring='f1_macro',
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    rs.fit(X_train, y_train)
    
    # Evaluation
    y_pred_temp = rs.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred_temp)
    
    print(f"\n{'='*20} {config['name']} RESULTS {'='*20}")
    print(f"✅ Best Params: {rs.best_params_}")
    print(f"📊 Test Accuracy: {acc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_temp, target_names=classes))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred_temp))
    print(f"{'='*60}\n")
    
    results_summary.append({'Model': config['name'], 'Accuracy': acc})
    
    if acc > best_global_score:
        best_global_score = acc
        best_global_model = rs.best_estimator_
        best_global_name = config['name']

# ==========================================
# 4. FINAL RESULTS & VISUALIZATION
# ==========================================
print("\n" + "*"*40)
print(f"🏆 WINNING MODEL: {best_global_name} (Score: {best_global_score:.4f})")
print("*"*40)

# Bar Plot
results_df = pd.DataFrame(results_summary)
plt.figure(figsize=(7, 5))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Model Comparison: Logistic Regression vs Random Forest')
plt.ylim(0.5, 1.0)
plt.show()

# ==========================================
# 5. FEATURE IMPORTANCE TABLES (THE MISSING PART)
# ==========================================
print(f"\n--- [ANALYSIS] WHAT DID THE {best_global_name.upper()} MODEL LEARN? ---")

try:
    # 1. Access the classifier and the TF-IDF feature names
    step_clf = best_global_model.named_steps['clf']
    tfidf_step = best_global_model.named_steps['features'].transformer_list[0][1]
    
    # 2. Get feature names and append 'Sentiment_Score' manually
    feature_names = list(tfidf_step.get_feature_names_out())
    feature_names.append("Sentiment_Score")
    
    # 3. Extract Importances based on model type
    df_importance = pd.DataFrame()
    
    if hasattr(step_clf, 'coef_'): # Logic for Logistic Regression
        # Coefficients: Negative -> Class 0 (Fake), Positive -> Class 1 (Real)
        importances = step_clf.coef_[0]
        df_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': importances})
        
        # Sort by actual value to separate Negative (Fake) and Positive (Real)
        df_importance = df_importance.sort_values(by='Coefficient')
        
        print("\n--- TABLE 1: TOP WORDS INDICATING 'FAKE' REVIEWS ---")
        print(df_importance.head(10).to_string(index=False))
        
        print("\n--- TABLE 2: TOP WORDS INDICATING 'REAL' REVIEWS ---")
        print(df_importance.tail(10).sort_values(by='Coefficient', ascending=False).to_string(index=False))

    elif hasattr(step_clf, 'feature_importances_'): # Logic for Random Forest
        # Feature Importances: Higher value -> More important (doesn't show direction)
        importances = step_clf.feature_importances_
        df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
        # Sort by importance
        df_importance = df_importance.sort_values(by='Importance', ascending=False)
        
        print("\n--- TABLE 1: MOST IMPORTANT FEATURES (Overall) ---")
        print(df_importance.head(15).to_string(index=False))

    # 4. Check Sentiment Score Impact
    print("\n--- TABLE 3: IMPACT OF SENTIMENT ANALYSIS ---")
    if 'Sentiment_Score' in df_importance['Feature'].values:
        print(df_importance[df_importance['Feature'] == 'Sentiment_Score'].to_string(index=False))
    else:
        print("Sentiment Score was filtered out during feature selection.")

except Exception as e:
    print(f"Could not generate feature tables: {e}")

# ==========================================
# 6. LIVE TEST
# ==========================================
print("\n" + "="*40)
print(f"      LIVE TEST MODE ({best_global_name})        ")
print("="*40)

while True:
    user_input = input("\nEnter a review (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    cleaned_input = clean_text(user_input)
    input_series = pd.Series([cleaned_input])
    
    # Prediction
    prediction_idx = best_global_model.predict(input_series)[0]
    raw_label = le.inverse_transform([prediction_idx])[0]
    proba = best_global_model.predict_proba(input_series).max()
    
    # Result String
    if raw_label == 'CG': 
        result_text = "FAKE (Computer Generated)"
    else:
        result_text = "REAL (Original)"
    
    print(f"Prediction: {result_text}")
    print(f"Confidence: {proba*100:.2f}%")