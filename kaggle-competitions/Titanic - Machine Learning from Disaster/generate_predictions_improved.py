#!/usr/bin/env python3
"""
Titanic Survival Prediction - IMPROVED Solution
Fixes data leakage, overfitting, and adds better features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*60)
print("TITANIC SURVIVAL PREDICTION - IMPROVED VERSION")
print("="*60)

# Load datasets
print("\n[1/9] Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Check survival distribution
print(f"\nSurvival rate in training data: {train['Survived'].mean():.2%}")

# Feature Engineering
print("\n[2/9] Feature Engineering...")
full_data = [train, test]

for dataset in full_data:
    # Extract Title from Name
    dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Create FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create IsAlone
    dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)

    # Extract Deck from Cabin
    dataset['Deck'] = dataset['Cabin'].str[0]
    dataset['Deck'] = dataset['Deck'].fillna('Unknown')

# Group rare titles
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print("Titles created:", train['Title'].unique())

# Add interaction features (IMPROVEMENT)
print("\n[3/9] Creating interaction features...")
for dataset in full_data:
    # Sex × Pclass interaction (very important for Titanic)
    dataset['Sex_Pclass'] = dataset['Sex'] + '_' + dataset['Pclass'].astype(str)

    # Title × Pclass interaction
    dataset['Title_Pclass'] = dataset['Title'] + '_' + dataset['Pclass'].astype(str)

# Missing Value Imputation
print("\n[4/9] Handling missing values...")

# Fill missing Age based on Pclass and Sex
for dataset in full_data:
    for pclass in [1, 2, 3]:
        for sex in ['male', 'female']:
            median_age = train[(train['Pclass'] == pclass) & (train['Sex'] == sex)]['Age'].median()
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Pclass'] == pclass) & (dataset['Sex'] == sex), 'Age'] = median_age

# Fill missing Embarked with mode
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(train['Embarked'].mode()[0])

# Fill missing Fare with median
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

print("Missing values after imputation:")
print(f"  Train - Age: {train['Age'].isnull().sum()}, Embarked: {train['Embarked'].isnull().sum()}, Fare: {train['Fare'].isnull().sum()}")
print(f"  Test  - Age: {test['Age'].isnull().sum()}, Embarked: {test['Embarked'].isnull().sum()}, Fare: {test['Fare'].isnull().sum()}")

# Create Age bins (using consistent bins across train/test)
print("\n[5/9] Creating binned features (FIX: No data leakage)...")

# Age groups - fixed bins
for dataset in full_data:
    dataset['AgeGroup'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 100],
                                  labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# FIXED: Fare groups - compute bins from TRAINING data only
_, fare_bins = pd.qcut(train['Fare'], q=4, retbins=True, duplicates='drop')
print(f"  Fare bins (from training): {fare_bins}")

for dataset in full_data:
    num_bins = len(fare_bins) - 1
    labels = ['Low', 'Medium', 'High', 'VeryHigh'][:num_bins]
    dataset['FareGroup'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=labels,
                                   include_lowest=True, right=True)
    # Handle test fares outside training range
    if dataset['FareGroup'].isnull().any():
        dataset['FareGroup'] = dataset['FareGroup'].cat.add_categories('Unknown').fillna('Unknown')

# IMPROVEMENT: Remove redundant features (use binned versions only)
print("\n[6/9] Preparing features (removing redundancy)...")
feature_cols = [
    'Pclass', 'Sex', 'Embarked', 'Title', 'FamilySize', 'IsAlone',
    'AgeGroup', 'FareGroup', 'Deck',  # Removed Age and Fare (redundant with binned versions)
    'Sex_Pclass', 'Title_Pclass'  # Added interaction features
]

# One-hot encode categorical variables
X_train_encoded = pd.get_dummies(train[feature_cols], drop_first=True)
X_test_encoded = pd.get_dummies(test[feature_cols], drop_first=True)
y_train = train['Survived']

# Ensure train and test have same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# IMPROVEMENT: Scale numerical features (FamilySize)
print("\n[7/9] Scaling numerical features...")
numerical_features = ['FamilySize']

# Find indices of numerical features in encoded dataframe
num_feature_indices = [i for i, col in enumerate(X_train_encoded.columns) if col in numerical_features]

if num_feature_indices:
    scaler = StandardScaler()
    # Scale only numerical columns
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    X_train_scaled.iloc[:, num_feature_indices] = scaler.fit_transform(X_train_encoded.iloc[:, num_feature_indices])
    X_test_scaled.iloc[:, num_feature_indices] = scaler.transform(X_test_encoded.iloc[:, num_feature_indices])

    X_train = X_train_scaled
    X_test = X_test_scaled
else:
    X_train = X_train_encoded
    X_test = X_test_encoded

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Model Comparison
print("\n[8/9] Comparing models with cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hyperparameter Tuning (FIXED: Prevent overfitting)
print("\n[9/9] Hyperparameter tuning with overfitting prevention...")

# Random Forest tuning - FIXED: No unlimited depth, higher min_samples_leaf
print("  Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],  # Removed None to prevent overfitting
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 6],  # Increased from 1 to prevent overfitting
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train, y_train)
print(f"    Best RF params: {rf_grid.best_params_}")
print(f"    Best RF score: {rf_grid.best_score_:.4f}")

# Gradient Boosting tuning - FIXED: Higher min_samples_leaf
print("  Tuning Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],  # Removed 0.2, added 0.05
    'max_depth': [3, 4, 5],  # More conservative depths
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4],  # Increased from 1 to prevent overfitting
    'subsample': [0.8, 1.0]  # Added subsample for regularization
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
gb_grid.fit(X_train, y_train)
print(f"    Best GB params: {gb_grid.best_params_}")
print(f"    Best GB score: {gb_grid.best_score_:.4f}")

# IMPROVEMENT: Create ensemble with voting
print("\n  Creating ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_grid.best_estimator_),
        ('gb', gb_grid.best_estimator_),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))
    ],
    voting='soft'
)

ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
print(f"  Ensemble CV score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")

# Select final model - compare individual best vs ensemble
if ensemble_scores.mean() >= max(rf_grid.best_score_, gb_grid.best_score_):
    final_model = ensemble
    final_score = ensemble_scores.mean()
    model_name = "Voting Ensemble (RF + GB + LR)"
else:
    if rf_grid.best_score_ >= gb_grid.best_score_:
        final_model = rf_grid.best_estimator_
        final_score = rf_grid.best_score_
        model_name = "Random Forest"
    else:
        final_model = gb_grid.best_estimator_
        final_score = gb_grid.best_score_
        model_name = "Gradient Boosting"

print(f"\n{'='*60}")
print(f"  Selected Model: {model_name}")
print(f"  CV Accuracy: {final_score:.4f} ({final_score*100:.2f}%)")
print(f"{'='*60}")

# Feature Importance (if not ensemble)
if hasattr(final_model, 'feature_importances_'):
    print("\n  Top 15 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(15).iterrows():
        print(f"    {row['feature']:40s}: {row['importance']:.4f}")

# Generate Predictions
print("\n[FINAL] Generating predictions...")
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

print(f"  Total predictions: {len(predictions)}")
print(f"  Predicted survivors: {predictions.sum()} ({100*predictions.sum()/len(predictions):.1f}%)")
print(f"  Predicted deaths: {(1-predictions).sum()} ({100*(1-predictions).sum()/len(predictions):.1f}%)")

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission_improved.csv', index=False)

print("\n" + "="*60)
print("IMPROVED SUBMISSION FILE CREATED: submission_improved.csv")
print("="*60)
print(f"\nFile details:")
print(f"  Shape: {submission.shape}")
print(f"  Expected: (418, 2)")
print(f"  Valid: {submission.shape == (418, 2)}")
print(f"\nFirst 10 predictions:")
print(submission.head(10).to_string(index=False))
print(f"\n✓ Ready for Kaggle submission!")
print(f"✓ Expected accuracy: ~{final_score*100:.1f}%")
print(f"\nIMPROVEMENTS APPLIED:")
print(f"  ✓ Fixed Fare binning data leakage")
print(f"  ✓ Prevented overfitting in hyperparameters")
print(f"  ✓ Removed redundant features (Age/Fare)")
print(f"  ✓ Added feature scaling")
print(f"  ✓ Added interaction features (Sex×Pclass, Title×Pclass)")
print(f"  ✓ Ensemble model for better generalization")
