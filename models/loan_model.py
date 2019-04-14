import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data_path = "models/data/loan_data_set.csv"

# def predict():
df = pd.read_csv(data_path)
df = df.dropna()
df = df.replace({
	'Gender': {'Male': 0, 'Female': 1},
	'Education': {'Not Graduate': 0, 'Graduate': 1},
	'Self_Employed': {'No': 0, 'Yes': 1},
	'Married': {'No': 0, 'Yes': 1},
	'Loan_Status': {'N': 0, 'Y': 1}
	})
df.loc[df['Dependents'] == '3+', 'Dependents'] = 3
df.loc[:, 'Dependents'] = df['Dependents'].astype('int64')
df.head()

area_dummies = pd.get_dummies(df['Property_Area'])
df['Urban'] = area_dummies['Urban']
df['Rural'] = area_dummies['Rural']
df.head()

X_all = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Urban', 'Rural']]
y_all = df['Loan_Status']
y_all.head()

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

confusion_matrix(y_test, preds)

(precision_score(y_test, preds), recall_score(y_test, preds))

f1_score(y_test, preds)

def get_data():
	return [X_train, X_test]

def predict(x):
    return model.predict_proba(x)

def simple_predict(x):
	return model.predict(x)

def classes():
	model.classes_
