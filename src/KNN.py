import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.neighbors as sn
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import numpy as np
import pickle

train_data = pd.read_csv('../train.csv')
# 归一化
# test_data = pd.read_csv('../test.csv') / 255
# 二值化
one_zero = sp.Binarizer(threshold=0)
test_data = pd.read_csv('../test.csv')
test_data = one_zero.transform(test_data)
sample_data = pd.read_csv('../sample_submission.csv')
# print(train_data.info())
# print(train_data.head())
# print(test_data.info())
# print(test_data.head())
# print(sample_data.head())
# print(sample_data.info())
'''
# 查看图片分类情况
# data_y = data['label'].value_counts()
# data_y.plot(kind='bar')
for i in range(20):
	plt.figure()
	plt.subplot(4,5,i+1)
	picture = test_data.iloc[i,:].reshape((28,28))
	plt.imshow(picture,cmap='binary')
plt.show()
'''
#归一化
# train_data_x = train_data.iloc[:,1:] / 255
#二值化
train_data_x = train_data.iloc[:,1:]
train_data_x = one_zero.transform(train_data_x)
train_data_y = train_data.iloc[:,0]
# print(train_data_x.info())
# print(type(train_data_y))
#划分训练集和测试集
train_x, test_x, train_y, test_y = ms.train_test_split(train_data_x,train_data_y,test_size=0.15, random_state=4)
# # kNN模型
# n_neighbors = np.arange(3,8,2)
# model = sn.KNeighborsClassifier()
# train_score, test_score = ms.validation_curve(model,train_data_x,train_data_y,'n_neighbors',n_neighbors,cv=3)
# print(train_score.mean(axis=1))
# print(test_score.mean(axis=1))
model = sn.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)
pred_y = model.predict(test_x)
print(sm.confusion_matrix(test_y, pred_y))
print(sm.classification_report(test_y, pred_y))
# model.fit(train_data_x, train_data_y)
# # with open('./KNN_model1.pkl', 'wb') as f:
# # 	pickle.dump(model, f)
pred_y = model.predict(test_data)
result = pd.DataFrame({'ImageId':np.arange(1,len(test_data)+1),'label':pred_y})
result.to_csv('./predict7.csv',index=False)


# with open('./KNN_model1.pkl', 'rb') as f:
#     model = pickle.load(f)
# pred_y = model.predict(train_data_x)
# print(sm.confusion_matrix(train_data_y, pred_y))
# print(sm.classification_report(train_data_y, pred_y))