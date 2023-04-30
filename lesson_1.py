import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# załadowanie tabeli z pliku excel
df = pd.read_excel('data/data_1.xlsx', sheet_name='Arkusz1')

# wyświetlenie tabeli
print(df)

le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])
print(df)


scaler = StandardScaler()
df[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4']] = scaler.fit_transform(
    df[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4']])
print(df)

df = pd.get_dummies(data=df, drop_first=True)
print(df)

print('Data:')
data = df[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4']]
print(data)


print('Target:')
target = df['Result']
print(target)

print('All data:')
all_data = np.c_[data, target]
print(all_data[:5])

df = pd.DataFrame(data=all_data, columns=[
                  'Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4'] + ['target'])
print(df)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy * 100))


print(y_pred)
