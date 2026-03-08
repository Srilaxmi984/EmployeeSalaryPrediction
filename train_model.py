import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("dataset/Salary_Data.csv")

# Clean column names
data.columns = data.columns.str.replace(" ", "_")

# Remove missing values
data = data.dropna()

# Encoders
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

# Encode categorical columns
data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Education_Level'] = le_education.fit_transform(data['Education_Level'])
data['Job_Title'] = le_job.fit_transform(data['Job_Title'])

# Features
X = data[['Age','Gender','Education_Level','Job_Title','Years_of_Experience']]

# Target
y = data['Salary']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200)

# Train model
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)

print("Model Accuracy:", score)

# Save model
pickle.dump(model, open("salary_model.pkl","wb"))

# Save encoders
pickle.dump((le_gender,le_education,le_job), open("encoders.pkl","wb"))

print("Model saved successfully")