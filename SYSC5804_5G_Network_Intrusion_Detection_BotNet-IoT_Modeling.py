#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import time


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


new_names = ['ID','1','2','3','4','5','6','7','8','9','10',
             '11','12','13','14','15','16','17','18','19','20','21','22','23','Label']

df = pd.read_csv(
    'D:/Network Intrusion 5G/BotNeTIoT-L01_label_NoDuplicates.csv', 
    names=new_names,           # Rename columns
    header=0,                  # Drop the existing header row
    usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],       # Read the first 5 columns
)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe


# ## Data Prepration

# In[ ]:


# Splitting lables
Y = df.Label
X = df.drop(['Label',], axis=1)


# In[ ]:


# Scaling
scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(X)
Print('Scaled Features:', X_scaled)


# In[ ]:


# Split test and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# # Modeling

# ## Gaussian Naive Bayes

# In[ ]:


model1 = GaussianNB()


# In[ ]:


# Training
start_time = time.time()
model1.fit(X_train, Y_train.values.ravel())
end_time = time.time()
#print("Training time: ", end_time-start_time)
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


# Predicting
start_time = time.time()
Y_test_pred1 = model1.predict(X_test)
end_time = time.time()
#print("Testing time: ",end_time-start_time)
print("Testing time: {:.2f}".format(end_time-start_time))


# In[ ]:


collections.Counter(Y_test_pred1)


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred1))
print()
classification = metrics.classification_report(Y_test, Y_test_pred1)
print("GNB Classification report:" "\n", classification)


# ## Random Forest

# In[ ]:


model2 = RandomForestClassifier(n_estimators=30)


# In[ ]:


# Training

start_time = time.time()
model2.fit(X_train, Y_train.values.ravel())
end_time = time.time()
#print("Training time: ",end_time-start_time)
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


# Prediction (Testing)

start_time = time.time()
Y_test_pred2 = model2.predict(X_test)
end_time = time.time()
#print("Testing time: ",end_time-start_time)
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred2))
print()
classification = metrics.classification_report(Y_test, Y_test_pred2)
print("RF Classification report:" "\n", classification)


# ## Support Vectot Machine

# In[ ]:


model3 = SVC(gamma = 'scale')


# In[ ]:


# Training

start_time = time.time()
model3.fit(X_train, Y_train.values.ravel())
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


# Prediction (Testing)

start_time = time.time()
Y_test_pred3 = model3.predict(X_test)
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred3))
print()
classification = metrics.classification_report(Y_test, Y_test_pred3)
print("SVM Classification report:" "\n", classification)


# ## Logistic Regression

# In[ ]:


model4 = LogisticRegression(max_iter=1200000)


# In[ ]:


# Training

start_time = time.time()
model4.fit(X_train, Y_train.values.ravel())
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


# Prediction (Testing)

start_time = time.time()
Y_test_pred4 = model4.predict(X_test)
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred4))
print()
classification = metrics.classification_report(Y_test, Y_test_pred4)
print("LR Classification report:" "\n", classification)


# ## Decision Tree

# In[ ]:


model5 = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)


# In[ ]:


# Training

start_time = time.time()
model5.fit(X_train, Y_train.values.ravel())
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


# # Prediction (Testing)

start_time = time.time()
Y_test_pred5 = model5.predict(X_test)
end_time = time.time()
print("Testinng time: {:.2f}".format(end_time-start_time))


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred5))
print()
classification = metrics.classification_report(Y_test, Y_test_pred5)
print("DT Classification report:" "\n", classification)


# ## K-nearest Neighbor

# In[ ]:


model6 = KNeighborsClassifier()


# In[ ]:


# Training

start_time = time.time()
model6.fit(X_train, Y_train.values.ravel())
end_time = time.time()
print("Training time: {:.2f}".format(end_time-start_time))


# In[ ]:


start_time = time.time()
Y_test_pred6 = model6.predict(X_test)
end_time = time.time()
print("Testinng time: {:.2f}".format(end_time-start_time))


# In[ ]:


print('Model accuracy:', accuracy_score(Y_test, Y_test_pred6))
print()
classification = metrics.classification_report(Y_test, Y_test_pred6)
print("KNN Classification report:" "\n", classification)


# # 1-Click to Run

# ## Note: mind the time when using this method!

# In[ ]:


from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train); 

# Train LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train);

# Train Gaussian Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)
            
# Train Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)

# Train SVM
SVM_Classifier = SVC(gamma = 'scale')
SVM_Classifier.fit(X_train, Y_train)

# Train RF
RF_Classifier = RandomForestClassifier(n_estimators=30)
RF_Classifier.fit(X_train, Y_train)


# In[ ]:


from sklearn import metrics

models = []
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))
models.append(('Support Vector Machine', SVM_Classifier))
models.append(('Random Forest', RF_Classifier))

for i, m in models:
    scores = cross_val_score(m, X_test, Y_test, cv=10)
    accuracy = metrics.accuracy_score(Y_test, m.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, m.predict(X_test))
    classification = metrics.classification_report(Y_test, m.predict(X_test))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()

