import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os

# Scikit-learn Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

if not os.path.exists('project_results'):
    os.makedirs('project_results')

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def calculate_sentiment_scores(x):
    return x.apply(get_sentiment).values.reshape(-1, 1)

def plot_feature_importance(model, feature_names, model_name):
    importances = []
    if hasattr(model, 'coef_'): 
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'): 
        importances = model.feature_importances_
    
    if len(importances) == len(feature_names):
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp['AbsImportance'] = df_imp['Importance'].abs()
        df_top = df_imp.sort_values(by='AbsImportance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=df_top, palette='viridis')
        plt.title(f'Top 15 Most Important Features - {model_name}')
        plt.tight_layout()
        filename = f"project_results/feature_importance_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"[INFO] Feature Importance Plot saved to {filename}")
    else:
        print("Feature mismatch, skipping plot.")

# --- YENİ EKLENEN LEARNING CURVE FONKSİYONU ---
def plot_learning_curve_graph(estimator, X, y, model_name):
    print(f"\n--- GENERATING LEARNING CURVE FOR {model_name} ---")
    print("(This may take a moment as it retrains the model on different subset sizes...)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, 
        train_sizes=np.linspace(0.1, 1.0, 5), # Verinin %10, %32, %55, %77 ve %100'ü ile dener
        scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training Examples (Data Size)")
    plt.ylabel("Score (Accuracy)")
    plt.grid(True)

    # Train Curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")

    # CV Curve
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")

    plt.legend(loc="best")
    filename = "project_results/learning_curve.png"
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    # plt.show()

# ==========================================
# 2. DATA LOADING
# ==========================================
print("--- 1. DATASET DESCRIPTION ---")
try:
    df = pd.read_csv("fake reviews dataset.csv", encoding="utf-8")
    print(f"Total Number of Samples: {len(df)}")
except FileNotFoundError:
    print("ERROR: 'fake reviews dataset.csv' not found.")
    exit()

# ==========================================
# 3. PRE-PROCESSING
# ==========================================
print("\n--- 2. PRE-PROCESSING ---")
df['text_cleaned'] = df['text_'].apply(clean_text)

le = LabelEncoder()
y = le.fit_transform(df['label'])
classes = list(le.classes_)
print(f"Classes Encoded: {classes}")

# ==========================================
# 4. DATA SPLITTING
# ==========================================
print("\n--- 3. DATA SPLITTING ---")
X = df['text_cleaned'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# ==========================================
# 5. MODELING & COMPARISON
# ==========================================
print("\n" + "="*50)
print("--- 4. MACHINE LEARNING MODELS ---")
print("="*50)

combined_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('sentiment', FunctionTransformer(calculate_sentiment_scores, validate=False))
])

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
        'params': {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, 20]}
    },
    {
        'name': 'Gradient Boosting',
        'pipeline': ImbPipeline([
            ('features', combined_features),
            ('smote', SMOTE(random_state=42)),
            ('clf', GradientBoostingClassifier(random_state=42))
        ]),
        'params': {
            'clf__n_estimators': [50, 100], 
            'clf__learning_rate': [0.1, 0.2],
            'clf__max_depth': [3, 5]
        }
    }
]

best_global_score = 0
best_global_model = None
best_global_name = ""
results_summary = [] 

for config in model_configs:
    print(f"\n⏳ TRAINING: {config['name']} with 5-Fold CV...")
    
    rs = RandomizedSearchCV(
        config['pipeline'], param_distributions=config['params'],
        n_iter=3, cv=5, scoring='f1_macro', n_jobs=1, random_state=42, verbose=1
    )
    rs.fit(X_train, y_train)
    
    y_train_pred = rs.best_estimator_.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    y_pred = rs.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"📊 {config['name']} Performance Analysis:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    
    diff = train_acc - test_acc
    if diff > 0.10: 
        print(f"   ⚠️ WARNING: High Overfitting Risk! (Gap: {diff:.2f})")
    else:
        print(f"   ✅ Model seems Balanced (Gap: {diff:.2f})")

    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion Matrix Save
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {config["name"]}')
    plt.savefig(f'project_results/confusion_matrix_{config["name"].replace(" ", "_")}.png')
    plt.close() 
    
    results_summary.append({
        'Model': config['name'], 
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc
    })
    
    if test_acc > best_global_score:
        best_global_score = test_acc
        best_global_model = rs.best_estimator_
        best_global_name = config['name']

# ==========================================
# 6. FINAL RESULTS & PLOTS
# ==========================================
print("\n" + "*"*40)
print(f"🏆 WINNER MODEL: {best_global_name} (Score: {best_global_score:.4f})")
print("*"*40)

# 1. Overfitting Bar Plot
df_res = pd.DataFrame(results_summary)
df_melted = df_res.melt(id_vars="Model", var_name="Dataset", value_name="Accuracy")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=df_melted, palette='muted')
plt.title('Train vs Test Accuracy (Overfitting Check)')
plt.ylim(0.5, 1.05)
plt.savefig('project_results/overfitting_analysis.png')
print("✅ Saved: project_results/overfitting_analysis.png")

# 2. Learning Curve (Sadece Kazanan Model İçin)
# Bu işlem biraz uzun sürebilir çünkü modeli 5-6 kere tekrar eğitir
try:
    plot_learning_curve_graph(best_global_model, X_train, y_train, best_global_name)
except Exception as e:
    print(f"Could not plot learning curve: {e}")

# 3. Feature Importance
print(f"\n--- GENERATING FEATURE IMPORTANCE FOR REPORT ({best_global_name}) ---")
try:
    step_clf = best_global_model.named_steps['clf']
    features_step = best_global_model.named_steps['features']
    tfidf_transformer = features_step.transformer_list[0][1]
    
    feature_names = list(tfidf_transformer.get_feature_names_out())
    feature_names.append("Sentiment_Score")
    
    plot_feature_importance(step_clf, feature_names, best_global_name)
except Exception as e:
    print(f"Could not plot features: {e}")

# ==========================================
# 7. LIVE TEST MODE
# ==========================================
print("\n" + "="*40)
print(f"      LIVE TEST MODE ({best_global_name})        ")
print("="*40)

while True:
    user_input = input("\nEnter a review (or 'q' to quit): ")
    if user_input.lower() == 'q': break
    
    input_series = pd.Series([clean_text(user_input)])
    prediction_idx = best_global_model.predict(input_series)[0]
    raw_label = le.inverse_transform([prediction_idx])[0]
    
    try:
        proba = best_global_model.predict_proba(input_series).max()
        conf_str = f"(Confidence: {proba*100:.2f}%)"
    except:
        conf_str = ""
    
    label = "FAKE (Computer Generated)" if raw_label == 'CG' else "REAL (Original)"
    print(f"Prediction: {label} {conf_str}")