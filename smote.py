# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:43:31 2025

@author: acojh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load your dataset
# X = ...  # Feature matrix
# y = ...  # Target labels (imbalanced)
# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
data = pd.read_csv('tweet_data.csv')
X = data.drop('Misinformation',axis=1)
y = data['Misinformation']

# Check class distribution
print("Original class distribution:", Counter(y))

count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(count_class.index, ['Low', 'No', 'Moderate', 'High', 'Severe'])
plt.show()

### 1. Oversampling (SMOTE) ###
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
X_smote, y_smote = smote.fit_resample(X, y)
print("After SMOTE (oversampling) class distribution :", Counter(y_smote))

count_class = y_smote.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.show()

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

print("Train set class distribution:", Counter(y_train))

count_class = y_train.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
addlabels(count_class.index, count_class.values)
plt.show()

svm_model = SVC()
#svm_model = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(4, 5), random_state=1, solver='lbfgs')
scores = cross_val_score(svm_model, X_train, y_train, cv=10)
print(f"Cross Validation Score:{np.mean(scores)*100:.2f}%")
svm_model.fit(X_train,y_train)

pred=svm_model.predict(X_train)
print(f"Traing Accuracy: {accuracy_score(y_train,pred)*100:.2f}%")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

"""
X_resampled = pd.DataFrame(X_smote)
y_resampled = pd.DataFrame(y_smote)
resampled.combine(X-resampled, y_resampled)
resampled.to_csv('smote_tweet_data.csv' , index=False)
"""
#X_resampled = pd.DataFrame(X_smote)
#y_resampled = pd.DataFrame(y_smote)
#X_resampled.to_csv('smote_tweet_data_features.csv', index=False)
#y_resampled.to_csv('smote_tweet_data_target.csv', index=False)

#save the classification model
#with open('svm_model_pkl', 'wb') as files:
    #pickle.dump(svm_model, files)
