#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', \
                 'bonus', \
                 'total_stock_value', \
                 'to_messages', \
                 'from_poi_to_this_person', 'from_messages', \
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print data_dict

### Task 2: Remove outliers
import matplotlib.pyplot as plt
import numpy as np

scratch = []
both = 0
salary_only = 0
email_only = 0
neither = 0
for person in data_dict:
    if data_dict[person].get("salary") == 'NaN':
        if data_dict[person].get("to_messages") == 'NaN':
            neither += 1
        else:
            email_only += 1
    else:
        if data_dict[person].get("to_messages") == 'NaN':
            salary_only += 1
        else:
            both += 1
            if ((data_dict[person].get("total_payments") < 10000000) and \
                (data_dict[person].get("to_messages") < 10000) and \
                (data_dict[person].get("from_messages") < 10000)):
                scratch.append([person, data_dict[person].get("poi"), \
                   data_dict[person].get("salary"), \
                   data_dict[person].get("total_payments"), \
                   data_dict[person].get("bonus"), \
                   data_dict[person].get("total_stock_value"), \
                   data_dict[person].get("to_messages"),  \
                   data_dict[person].get("from_poi_to_this_person"), \
                   data_dict[person].get("from_messages"), \
                   data_dict[person].get("from_this_person_to_poi"), \
                   data_dict[person].get("shared_receipt_with_poi")])
print scratch
print "total people: ", len(data_dict)
print "people with salary + email data: ", both
print "people with only salary data: ", salary_only
print "people with only email data: ", email_only
print "people with neither: ", neither
print "total people after cleaning outliers: ", len(scratch)
##scratch2 = []
##for i in scratch:
##    scratch2.append(i[3])
##    if i[3] > 10000000:
##        print i
##plt.boxplot(scratch2)
##plt.show()

### Task 3: Create new feature(s)
new_feature = 0
m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in scratch:
    new_feature = float((float(i[7])+float(i[9])+float(i[10]))/(float(i[6])+float(i[8])))
    i.append(new_feature)
    if i[1] > m[0]: m[0] = float(i[1])
    if i[2] > m[1]: m[1] = float(i[2])
    if i[3] > m[2]: m[2] = float(i[3])
    if ((i[4] > m[3]) and (i[4] <> 'NaN')): m[3] = float(i[4])
    if ((i[5] > m[4]) and (i[5] <> 'NaN')): m[4] = float(i[5])
    if i[6] > m[5]: m[5] = float(i[6])
    if i[7] > m[6]: m[6] = float(i[7])
    if i[8] > m[7]: m[7] = float(i[8])
    if i[9] > m[8]: m[8] = float(i[9])
    if i[10] > m[9]: m[9] = float(i[10])
    if i[11] > m[10]: m[10] = float(i[11])    
print scratch
print m
##print len(scratch)

features_list.append('poi_email_ratio')
##print features_list
cleaned_names = []
for i in scratch:
    cleaned_names.append(i[0])
##print cleaned_names

### Store to my_dataset for easy export below.
my_dataset = {}
for i in data_dict:
    if i in cleaned_names:
        my_dataset[i] = {}
        my_dataset[i]['poi'] = data_dict[i]['poi']
        my_dataset[i]['salary'] = data_dict[i]['salary']/m[1]
        my_dataset[i]['total_payments'] = data_dict[i]['total_payments']/m[2]
        if data_dict[i]['bonus'] == 'NaN':
            my_dataset[i]['bonus'] = 0
        else:
            my_dataset[i]['bonus'] = data_dict[i]['bonus']/m[3]
        if data_dict[i]['total_stock_value'] == 'NaN':
            my_dataset[i]['total_stock_value'] = 0
        else:
            my_dataset[i]['total_stock_value'] = data_dict[i]['total_stock_value']/m[4]
        my_dataset[i]['to_messages'] = data_dict[i]['to_messages']/m[5]
        my_dataset[i]['from_poi_to_this_person'] = data_dict[i]['from_poi_to_this_person']/m[6]
        my_dataset[i]['from_messages'] = data_dict[i]['from_messages']/m[7]
        my_dataset[i]['from_this_person_to_poi'] = data_dict[i]['from_this_person_to_poi']/m[8]
        my_dataset[i]['shared_receipt_with_poi'] = data_dict[i]['shared_receipt_with_poi']/m[9]
        my_dataset[i]['poi_email_ratio'] = \
        float(((float(data_dict[i]['from_poi_to_this_person'])\
               +float(data_dict[i]['from_this_person_to_poi'])\
               +float(data_dict[i]['shared_receipt_with_poi']))/\
              (float(data_dict[i]['to_messages'])+float(data_dict[i]['from_messages']))))/m[10]

print my_dataset
print len(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print labels
print features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB

estimators = [('pca', PCA()), ('svm', SVC())]
pipe = Pipeline(estimators)

params = dict(pca__n_components=[2, 5, 10], svm__C=[0.1, 1, 10, 100])
clf = GridSearchCV(pipe, param_grid=params)

svm = SVC()
gnb = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.7, random_state=42)
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

print classification_report(labels_test, labels_pred)
print confusion_matrix(labels_test, labels_pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)