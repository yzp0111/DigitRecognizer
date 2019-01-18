import pandas as pd
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import numpy as np

train_data = pd.read_csv('../train.csv')
test_data = pd.read_csv('../test.csv')
train_data_y = train_data['label'].values
print(type(train_data_y))
train_data_x = train_data.drop('label',axis=1)
print(type(train_data_x))
one_zero = sp.Binarizer(threshold=0)
test_data = one_zero.transform(test_data)
train_data_x = one_zero.transform(train_data_x)
train_x, test_x, train_y, test_y = ms.train_test_split(train_data_x,train_data_y, test_size=0.15,random_state=4)
# params = [{'max_depth':[35,40], 'n_estimators':[1600,1800]}]
# # model = ms.GridSearchCV(se.RandomForestClassifier(random_state=4), params, cv=3)
# model.fit(train_x, train_y)
# for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
#     print(param, score)
# print(model.best_params_)
# print(model.best_score_)
# print(model.best_estimator_)
model = se.RandomForestClassifier(max_depth=35, n_estimators=1800,random_state=3)
model.fit(train_x,train_y)
pred_y = model.predict(test_x)
print(sm.confusion_matrix(test_y,pred_y))
print(sm.classification_report(test_y,pred_y))
pred_y = model.predict(test_data)
result = pd.DataFrame({'ImageId':np.arange(1,len(test_data)+1),'label':pred_y})
result.to_csv('./predict_RF.csv',index=False)