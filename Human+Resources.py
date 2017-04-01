
# coding: utf-8

# # Human Resources Notebook

# ## Data Setup

# Read datafile into memory

# In[1]:

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv('data/HR_comma_sep.csv')


# In[2]:

print(df.head())


# Clean up data and get ready for ML

# In[3]:

# vectorize string columns
vectorized_sales = pd.get_dummies(df['sales'])
vectorized_salary = pd.get_dummies(df['salary'])

# add vectorized columns to the original dataframe and drop the old columns
df = pd.concat([df, vectorized_salary, vectorized_sales], axis=1)
df.drop('sales', axis=1, inplace=True)
df.drop('salary', axis=1, inplace=True)

# move the Y column (left) to the end of the dataframe for readability
y = df['left']
df.drop('left', axis=1, inplace=True)
df = pd.concat([df, y], axis=1)


# View the data inside the transformed dataframe

# In[4]:

print(df.head())


# In[5]:

from sklearn.model_selection import train_test_split

# split up the training and test data before we start doing anything so we don't generate bias in the experiments
X_train, X_test, y_train, y_test = train_test_split(df.ix[:,:-1], df.ix[:,-1:], test_size=0.3, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

X_train.shape
X_test.shape
y_train.shape


# ## Algorithms

# ### Decision Trees

# In[6]:

from sklearn import tree
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'max_depth': [1, 10, 25, 50, 75], 'max_leaf_nodes': [2, 10, 15, 25, 50, 75]}]
dt_clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=10)
dt_clf.fit(X_train, y_train)
dt_optimized = dt_clf.best_estimator_
print(dt_clf.best_params_)


# In[7]:

dt_optimized.fit(X_train, y_train)
dt_optimized.score(X_test, y_test)


# In[8]:

from sklearn.model_selection import learning_curve
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt

plot_learning_curve(dt_optimized, title='Decision Tree learning curve', X=X_train, y=y_train, cv=10)
plt.show()


# ## knn

# In[22]:

from sklearn.neighbors import KNeighborsClassifier

tuned_parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [1, 2, 5, 10, 25]}]
knn_clf = GridSearchCV(KNeighborsClassifier(n_neighbors=1), tuned_parameters, cv=10)
knn_clf.fit(X_train, y_train)
knn_optimized = knn_clf.best_estimator_
print(knn_clf.best_params_)


# In[23]:

knn_optimized.fit(X_train, y_train)
knn_optimized.score(X_test, y_test)


# In[34]:

plot_learning_curve(knn_optimized, title='Knn learning curve', X=X_train, y=y_train, cv=10)
plt.show()


# ## Support Vector Machines

# In[17]:

from sklearn.svm import SVC

tuned_parameters = [{'C': [1, 10, 100, 1000]}]
svm_clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10)
svm_clf.fit(X_train, y_train)
svm_optimized = svm_clf.best_estimator_
print(svm_clf.best_params_)


# In[18]:

svm_optimized.fit(X_train, y_train)
svm_optimized.score(X_test, y_test)


# In[19]:

plot_learning_curve(svm_optimized, title='SVM learning curve', X=X_train, y=y_train, cv=10)
plt.show()


# ### Neural Nets

# In[13]:

from sklearn.neural_network import MLPClassifier

tuned_parameters = [{'hidden_layer_sizes': [(3, 2), (4, 2), (5, 2), (6, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 2)]}]
ann_clf = GridSearchCV(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), tuned_parameters, cv=10)
ann_clf.fit(X_train, y_train)
ann_optimized = ann_clf.best_estimator_
print(ann_clf.best_params_)


# In[15]:

ann_optimized.fit(X_train, y_train)
ann_optimized.score(X_test, y_test)


# In[16]:

plot_learning_curve(ann_optimized, title='Ann learning curve', X=X_train, y=y_train, cv=10)
plt.show()


# ### Boosting

# In[10]:

from sklearn.ensemble import AdaBoostClassifier

tuned_parameters = [{'n_estimators': [50, 100, 150, 200]}]
boost_clf = GridSearchCV(AdaBoostClassifier(n_estimators=100), tuned_parameters, cv=10)
boost_clf.fit(X_train, y_train)
boost_optimized = boost_clf.best_estimator_
print(boost_clf.best_params_)


# In[11]:

boost_optimized.fit(X_train, y_train)
boost_optimized.score(X_test, y_test)


# In[12]:

plot_learning_curve(boost_optimized, title='Boost learning curve', X=X_train, y=y_train, cv=10)
plt.show()


# ## Analysis

# ### Performance analysis

# In[24]:

scores = []
scores.append(dt_optimized.score(X_test, y_test))
scores.append(knn_optimized.score(X_test, y_test))
scores.append(svm_optimized.score(X_test, y_test))
scores.append(ann_optimized.score(X_test, y_test))
scores.append(boost_optimized.score(X_test, y_test))

scores_df = pd.DataFrame(scores, index=['Decision Trees', 'KNN', 'SVM', 'ANN', 'Boosting'])
ax = scores_df.plot(kind='bar', legend=False)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Accuracy')
plt.show()


# ### Time analysis

# In[25]:

import time 

start_dt_fit = time.time()
dt_optimized.fit(X_train, y_train)
end_dt_fit = time.time()
dt_optimized.score(X_test, y_test)
end_dt_query = time.time()

start_knn_fit = time.time()
knn_optimized.fit(X_train, y_train)
end_knn_fit = time.time()
knn_optimized.score(X_test, y_test)
end_knn_query = time.time()

start_svm_fit = time.time()
svm_optimized.fit(X_train, y_train)
end_svm_fit = time.time()
svm_optimized.score(X_test, y_test)
end_svm_query = time.time()

start_ann_fit = time.time()
ann_optimized.fit(X_train, y_train)
end_ann_fit = time.time()
ann_optimized.score(X_test, y_test)
end_ann_query = time.time()

start_boost_fit = time.time()
boost_optimized.fit(X_train, y_train)
end_boost_fit = time.time()
boost_optimized.score(X_test, y_test)
end_boost_query = time.time()


# In[26]:

times = []
times.append([(end_dt_fit - start_dt_fit), (end_dt_query - end_dt_fit)])
times.append([(end_knn_fit - start_knn_fit), (end_knn_query - end_knn_fit)])
times.append([(end_svm_fit - start_svm_fit), (end_svm_query - end_svm_fit)])
times.append([(end_ann_fit - start_ann_fit), (end_ann_query - end_ann_fit)])
times.append([(end_boost_fit - start_boost_fit), (end_boost_query - end_boost_fit)])

times_df = pd.DataFrame(times, index=['Decision Trees', 'KNN', 'SVM', 'ANN', 'Boosting'], columns=['Training Time', 'Query Time'])
ax = times_df.plot(kind='bar')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Time (in milliseconds)')
plt.show()

