#!/usr/bin/env python3
"""
Silver Dataset ML Analysis
Train models on the HumanEval silver dataset to compare with Task 1 Part B results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def analyze_silver_dataset():
    """Analyze the silver dataset and train models"""
    print("🎯 Silver Dataset ML Analysis")
    print("=" * 50)
    
    # Load silver dataset
    df = pd.read_csv('data/humaneval_silver_dataset_final.csv')
    print(f"📊 Dataset loaded: {len(df):,} examples")
    print(f"🏷️ Class distribution:")
    print(df['Class'].value_counts())
    
    # Prepare features
    def prepare_features(df):
        """Combine comments and code for feature extraction"""
        combined_text = df['Comments'].astype(str) + ' ' + df['Surrounding Code Context'].astype(str)
        return combined_text, df['Class']
    
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(X_train):,} samples")
    print(f"  Testing: {len(X_test):,} samples")
    
    # Models to test
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    results = {}
    
    print(f"\n🤖 Training models...")
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='useful')
        
        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"    ✅ Accuracy: {accuracy:.4f}")
        print(f"    ✅ F1 Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"    📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['not useful', 'useful']))
    
    # Summary
    print(f"\n🏆 SILVER DATASET RESULTS SUMMARY:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Compare with Task 1 Part B results
    print(f"\n📈 COMPARISON WITH TASK 1 PART B:")
    print("=" * 50)
    print("Silver Dataset (HumanEval Python):")
    best_f1 = max(results[model]['f1_score'] for model in results)
    print(f"  Best F1 Score: {best_f1:.4f}")
    print("Task 1 Part B (C code + synthetic):")
    print(f"  Previous F1 Score: 0.8501")
    
    improvement = ((best_f1 - 0.8501) / 0.8501) * 100 if best_f1 > 0.8501 else ((best_f1 - 0.8501) / 0.8501) * 100
    print(f"  Difference: {improvement:+.2f}%")
    
    if best_f1 > 0.85:
        print("🎉 Silver dataset shows competitive performance!")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Model': model_name,
            'Dataset': 'Silver_HumanEval',
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1_Score': f"{metrics['f1_score']:.4f}",
            'Language': 'Python',
            'Source': 'HumanEval_Silver'
        }
        for model_name, metrics in results.items()
    ])
    
    results_df.to_csv('silver_dataset_results.csv', index=False)
    print(f"\n💾 Results saved to: silver_dataset_results.csv")
    
    return results

if __name__ == "__main__":
    analyze_silver_dataset()