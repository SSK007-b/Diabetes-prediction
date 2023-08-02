import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv(r"D:\ML\Diabetic\diabetes_dataset1.csv")
X = dataset.iloc[: , :-1]
y = dataset.iloc[: , -1]

regressor = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)
regressor.fit(X , y)

joblib.dump(regressor , "diabModel.pkl")