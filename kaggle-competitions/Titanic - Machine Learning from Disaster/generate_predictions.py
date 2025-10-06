#!/usr/bin/env python3
"""
Titanic Survival Prediction - Complete Solution
Generates predictions for Kaggle submission
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*60)
print("TITANIC SURVIVAL PREDICTION")
print("="*60)

# Load datasets
print("\n[1/8] Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Check survival distribution
print(f"\nSurvival rate in training data: {train['Survived'].mean():.2%}")

# Feature Engineering
print("\n[2/8] Feature Engineering...")
full_data = [train, test]

for dataset in full_data:
    # Extract Title from Name
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Create FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create IsAlone
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

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

# Missing Value Imputation
print("\n[3/8] Handling missing values...")

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

# Create Age and Fare bins
print("\n[4/8] Creating binned features...")
for dataset in full_data:
    # Age groups
    dataset['AgeGroup'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # Fare groups (quartiles)
    dataset['FareGroup'] = pd.qcut(dataset['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')

# Prepare features for modeling
print("\n[5/8] Preparing features...")
feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']

# One-hot encode categorical variables
X_train = pd.get_dummies(train[feature_cols], drop_first=True)
X_test = pd.get_dummies(test[feature_cols], drop_first=True)
y_train = train['Survived']

# Ensure train and test have same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Model Comparison
print("\n[6/8] Comparing models with cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hyperparameter Tuning
print("\n[7/8] Hyperparameter tuning for best models...")

# Random Forest tuning
print("  Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
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
print(f"    Best RF score: {rf_grid.best_score_:.4f}")

# Gradient Boosting tuning
print("  Tuning Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
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
print(f"    Best GB score: {gb_grid.best_score_:.4f}")

# Select final model
if rf_grid.best_score_ >= gb_grid.best_score_:
    final_model = rf_grid.best_estimator_
    final_score = rf_grid.best_score_
    model_name = "Random Forest"
else:
    final_model = gb_grid.best_estimator_
    final_score = gb_grid.best_score_
    model_name = "Gradient Boosting"

print(f"\n  Selected Model: {model_name}")
print(f"  CV Accuracy: {final_score:.4f} ({final_score*100:.2f}%)")

# Feature Importance
print("\n  Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:30s}: {row['importance']:.4f}")

# Generate Predictions
print("\n[8/8] Generating predictions...")
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

submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION FILE CREATED: submission.csv")
print("="*60)
print(f"\nFile details:")
print(f"  Shape: {submission.shape}")
print(f"  Expected: (418, 2)")
print(f"  Valid: {submission.shape == (418, 2)}")
print(f"\nFirst 10 predictions:")
print(submission.head(10).to_string(index=False))
print(f"\n✓ Ready for Kaggle submission!")
print(f"✓ Expected accuracy on leaderboard: ~{final_score*100:.1f}%")
