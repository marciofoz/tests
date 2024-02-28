
'''
  Usando eXtreme Gradient Boost
  0,984 acc
'''
#from xgboost import XGBRFClassifier
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

xgb.set_config(verbosity=3)

data = pd.read_csv('IoTID20_preprocessada.csv')
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

labels = data['Label']
data = data.drop('Label', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20, random_state = 42)

# Instantiate the XGBRFClassifier
clf = xgb.XGBRFClassifier(n_estimators=1000, max_depth=100, subsample=0.8, colsample_bynode=0.2)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

p = clf.predict(X_test)

print(classification_report(y_test, p, digits=3))