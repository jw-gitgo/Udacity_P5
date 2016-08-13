#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', \
                 'total_payments', 'exercised_stock_options', 'bonus', \
                 'restricted_stock', 'shared_receipt_with_poi', \
                 'restricted_stock_deferred', 'total_stock_value', \
                 'expenses', 'loan_advances', 'from_messages', 'other', \
                 'from_this_person_to_poi', 'director_fees', \
                 'deferred_income', 'long_term_incentive', \
                 'from_poi_to_this_person']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print data_dict
print len(data_dict)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
print len(data_dict)
my_dataset = data_dict

### Task 3: Create new feature(s)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print labels
print features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
b = SelectKBest(f_classif, k=4)
features_new = b.fit_transform(features, labels)
print len(features_new)
print features_new
print b.scores_

    
m = [0, 0, 0, 0]
for i in my_dataset:
    if ((my_dataset[i]['salary'] > m[0]) and (my_dataset[i]['salary'] <> 'NaN')):
        m[0] = float(my_dataset[i]['salary'])
    if ((my_dataset[i]['exercised_stock_options'] > m[1]) and (my_dataset[i]['exercised_stock_options'] <> 'NaN')):
        m[1] = float(my_dataset[i]['exercised_stock_options'])
    if ((my_dataset[i]['bonus'] > m[2]) and (my_dataset[i]['bonus'] <> 'NaN')):
        m[2] = float(my_dataset[i]['bonus'])
    if ((my_dataset[i]['total_stock_value'] > m[3]) and (my_dataset[i]['total_stock_value'] <> 'NaN')):
        m[3] = float(my_dataset[i]['total_stock_value'])
print m
    
for i in my_dataset:
    if ((my_dataset[i]['salary'] <> 'NaN') and (my_dataset[i]['exercised_stock_options'] <> 'NaN')\
        and (my_dataset[i]['bonus'] <> 'NaN') and (my_dataset[i]['total_stock_value'] <> 'NaN')):
        my_dataset[i]['scaled_salary_bonus_stock'] = (float(my_dataset[i]['salary'])/m[0]) + \
        (float(my_dataset[i]['exercised_stock_options'])/m[1]) + \
        (float(my_dataset[i]['bonus'])/m[2]) + \
        (float(my_dataset[i]['total_stock_value'])/m[3])
    else:
        my_dataset[i]['scaled_salary_bonus_stock'] = 0
    
features_list.append('scaled_salary_bonus_stock')
print my_dataset

## Redo with the new feature
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

b = SelectKBest(f_classif, k=6)
features_new = b.fit_transform(features, labels)
print len(features_new)
print features_new
print b.scores_

##features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus', \
##                 'total_stock_value', 'scaled_salary_bonus_stock']
##data = featureFormat(my_dataset, features_list, sort_keys = True)
##labels, features = targetFeatureSplit(data)

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, precision_recall_fscore_support, make_scorer

score_function = make_scorer(f1_score)
##estimators = [('pca', PCA()), ('svm', SVC())]
##pipe = Pipeline(estimators)

svm = SVC()
gnb = GaussianNB()
adb = AdaBoostClassifier()
rmf = RandomForestClassifier()
##clf = RandomForestClassifier(min_samples_split=5, n_estimators=1, criterion='entropy', min_samples_leaf=3)
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME.R')


svm_params = {'C':[0.1, 1, 10, 100]}
gnb_params = {}
adb_params = {'n_estimators':[20, 50, 100], 'learning_rate':[0.5, 1, 5, 10], 'algorithm':('SAMME', 'SAMME.R')}
rmf_params = {'n_estimators':[1, 10, 20, 50, 100], 'criterion':('gini', 'entropy'), \
             'min_samples_split':[1, 2, 3, 5], 'min_samples_leaf':[1, 2, 3]}
##clf = GridSearchCV(adb, param_grid=adb_params, scoring=score_function)


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
from sklearn.cross_validation import StratifiedShuffleSplit

cv = \
    StratifiedShuffleSplit(labels, 1000, test_size=0.3, random_state=42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

##print clf.best_params_

print classification_report(labels_test, labels_pred)
print confusion_matrix(labels_test, labels_pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)