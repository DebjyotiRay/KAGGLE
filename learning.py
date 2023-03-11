import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

train_data = pd.read_csv('train.csv')
train_data.head()


test_data = pd.read_csv('train.csv')
test_data.head()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

y=train_data["Survived"]


features = ["Pclass", "Sex","SibSp","Parch"]

x=pd.get_dummies(train_data[features])
x_testing = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x,y)

predict = model.predict()
accuracy_score = accuracy_score(y,predict)
predictions = model.predict(x_testing)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
#Your submission was successfully saved!

