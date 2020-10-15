import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("diabetes_csv.csv")
df["class"] = df["class"].apply(lambda x: 1 if x == "tested_positive" else 0)
x = df[["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]]
y = df[["class"]]
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
model_1 = xgb.XGBClassifier()
model_2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
train_model_1 = model_1.fit(x_train, y_train)
train_model_2 = model_2.fit(x_train, y_train)
pred1 = train_model_1.predict(x_test)
pred2 = train_model_2.predict(x_test)
print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))
print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))
print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
param_test = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
g_search = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27), param_grid=param_test, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
train_model4 = g_search.fit(x_train, y_train)
pred4 = train_model4.predict(x_test)
print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred4) * 100))
param_test3 = {
 'gamma': [i/10.0 for i in range(0, 5)]
}
g_search3 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                     min_child_weight=6, gamma=0, subsample=0.8,
                                                     colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                                                     scale_pos_weight=1, seed=27),
                         param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

train_model6 = g_search3.fit(x_train, y_train)
pred6 = train_model6.predict(x_test)
print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred6) * 100))
rfc = RandomForestClassifier()
rfc_model = rfc.fit(x_train, y_train)
pred8 = rfc_model.predict(x_test)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))
