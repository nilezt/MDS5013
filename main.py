# Suppress warnings
#import warnings

#warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd

# Load the datasets
data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

''''df_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df_5050 = pd.read_csv(
    'diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Display the first few rows of each dataset
df_012.head()
print("df_012 before head:", df_012.head)
df_binary.head()
df_5050.head()
'''

#2. Explore the Data
# Display the first few rows
print(data.head())
print(data.isnull().sum())  # Check for missing values

#3. Define Features (X) and Target (y)
# Drop the target column and separate features
X = data.drop(columns=['Diabetes_012'])  # Features (all except 'Diabetes_012')
y = data['Diabetes_012']  # Target column (the diabetes status)


#4. Split the Data into Training and Test Sets
from sklearn.model_selection import train_test_split
# Split into train and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#5. Train the Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Create a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
# Train the model
rf_model.fit(X_train, y_train)


#6. Make Predictions
# Make predictions on the test data
y_pred = rf_model.predict(X_test)

#7. Evaluate Model Performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification report
class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")