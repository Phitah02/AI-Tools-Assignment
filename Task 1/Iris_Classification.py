# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and explore the dataset
print("Step 1: Loading and exploring the dataset...")
df = pd.read_csv("Task 1\Iris.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())
print("\nClass distribution:")
print(df['Species'].value_counts())

# Step 2: Data Preprocessing
print("\nStep 2: Data Preprocessing...")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop the 'Id' column as it's not useful for prediction
df = df.drop('Id', axis=1)
print("\nDropped 'Id' column")

# Encode the target variable (Species) from categorical to numerical
label_encoder = LabelEncoder()
df['Species_encoded'] = label_encoder.fit_transform(df['Species'])
print("\nLabel encoding mapping:")
for i, species in enumerate(label_encoder.classes_):
    print(f"{species}: {i}")

# Separate features and target variable
X = df.drop(['Species', 'Species_encoded'], axis=1)  # Features
y = df['Species_encoded']  # Target variable

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Features: {X.columns.tolist()}")

# Step 3: Split the data into training and testing sets
print("\nStep 3: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 70% training, 30% testing
    random_state=42,  # For reproducibility
    stratify=y  # Maintain class distribution in splits
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Testing set class distribution:\n{pd.Series(y_test).value_counts()}")

# Step 4: Train the Decision Tree Classifier
print("\nStep 4: Training Decision Tree Classifier...")
# Create decision tree classifier
dt_classifier = DecisionTreeClassifier(
    random_state=42,  # For reproducibility
    max_depth=3  # Limit tree depth to prevent overfitting
)

# Train the model
dt_classifier.fit(X_train, y_train)
print("Decision Tree classifier trained successfully!")

# Step 5: Make predictions
print("\nStep 5: Making predictions...")
y_pred = dt_classifier.predict(X_test)
y_pred_proba = dt_classifier.predict_proba(X_test)

print("First 5 predictions:")
for i in range(5):
    actual_species = label_encoder.inverse_transform([y_test.iloc[i]])[0]
    predicted_species = label_encoder.inverse_transform([y_pred[i]])[0]
    print(f"Sample {i+1}: Actual: {actual_species}, Predicted: {predicted_species}")

# Step 6: Evaluate the model
print("\nStep 6: Model Evaluation...")

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 7: Feature Importance
print("\nStep 7: Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Decision Tree Classifier')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Step 8: Example predictions on new data
print("\nStep 8: Example of making predictions on new data...")
# Create some example measurements
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Likely Iris-setosa
    [6.0, 2.7, 5.1, 1.6],  # Likely Iris-versicolor
    [7.2, 3.6, 6.1, 2.5]   # Likely Iris-virginica
])

new_predictions = dt_classifier.predict(new_samples)
new_probabilities = dt_classifier.predict_proba(new_samples)

print("Predictions for new samples:")
for i, (pred, probs) in enumerate(zip(new_predictions, new_probabilities)):
    species_name = label_encoder.inverse_transform([pred])[0]
    print(f"\nSample {i+1}: {new_samples[i]}")
    print(f"Predicted species: {species_name}")
    print("Probability distribution:")
    for j, prob in enumerate(probs):
        species = label_encoder.inverse_transform([j])[0]
        print(f"  {species}: {prob:.4f}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Model: Decision Tree Classifier")
print(f"Dataset: Iris Species (150 samples, 3 classes)")
print(f"Best performing feature: {feature_importance.iloc[0]['feature']}")
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Model is {'GOOD' if accuracy > 0.9 else 'NEEDS IMPROVEMENT'}")