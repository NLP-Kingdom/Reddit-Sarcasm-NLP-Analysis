import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, mean_squared_error, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

raw_df = pd.read_csv('train-balanced-sarcasm.csv')
raw_df.head()

## Drop Features ##

# Drop NAs
raw_df.dropna(inplace=True)

# Select 100000 rows of sample
# Reset index so the cross validation later won't go wrong
filter_df = raw_df.sample(n=100000, random_state=000).reset_index(drop=True)

# Drop irrelevant features
filter_df.drop(['author', 'date', 'created_utc'], axis=1)

## Categorical Process ##

# Instantiate the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform the processed comments and parent_comment
tfidf_comment = tfidf_vectorizer.fit_transform(filter_df['comment'])
tfidf_parent_comment = tfidf_vectorizer.fit_transform(filter_df['parent_comment'])

categorical_columns = ['subreddit']
for i in categorical_columns:
    filter_df = pd.concat([filter_df, pd.get_dummies(filter_df[i], drop_first=True, prefix=i)], axis=1)
    filter_df = filter_df.drop(i, axis=1)

## Split Dataset ##

# Y is the response variable
Y = filter_df['label']

# X is the features
X = tfidf_comment

# Split the data (Train 0.8, Test 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=000)

## K-Fold CV Setup ##

# Set up K-Fold Cross Validation
n_splits = 5
shuffle = True
random_state = 000
cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

## A Function to Create Dictionary ##

def create_dictionary(param_1,param_2):
    result_dictionary = {}
    for i in param_1:
        result_dictionary[i] = {}
        for j in param_2:
                result_dictionary[i][j] = {}
    return result_dictionary

## Random Forest ##

# CV in Trees
# Set Hyperparameter (Lambda) values to cross validate
max_depth = [2, 5, 10, 15, 20, 25]
number_of_trees = [51, 101, 151, 201]

cross_validate_result = create_dictionary(number_of_trees, max_depth)
cross_validate_recall = create_dictionary(number_of_trees, max_depth)
cross_validate_precision = create_dictionary(number_of_trees, max_depth)
cross_validate_mse = create_dictionary(number_of_trees, max_depth)

with open('Random Forest CV Summary.txt', 'w') as file:
    for tree in number_of_trees:
        for depth in max_depth:
            print('Depth of Tree : ', depth, ' Number of Trees ', tree)
            file.write(f'Depth of Tree : {depth}, Number of Trees: {tree}\n')

            accuracies = []
            recall_scores = []
            precision_scores = []
            mse_scores = []

            random_forest_cv = RandomForestClassifier(n_estimators=tree, max_depth=depth)

            for train_index, test_index in cv.split(X):
                # change to loc to define the rows in the dataframe
                X_cv_train, X_cv_test = X[train_index], X[test_index]
                Y_cv_train, Y_cv_test = Y[train_index], Y[test_index]

                random_forest_cv.fit(X_cv_train, Y_cv_train)
                Y_pred = random_forest_cv.predict(X_cv_test)

                # Cross-Validation Prediction Error
                score = random_forest_cv.score(X_cv_test, Y_cv_test)
                accuracies.append(score)
                recall_scores.append(recall_score(Y_cv_test, Y_pred))
                precision_scores.append(precision_score(Y_cv_test, Y_pred))
                mse_scores.append(mean_squared_error(Y_cv_test, Y_pred))

            cross_validate_result[tree][depth] = (sum(accuracies) / len(accuracies))
            cross_validate_recall[tree][depth] = (sum(recall_scores) / len(recall_scores))
            cross_validate_precision[tree][depth] = (sum(precision_scores) / len(precision_scores))
            cross_validate_mse[tree][depth] = (sum(mse_scores) / len(mse_scores))

            file.write(f"Accuracy: {np.mean(accuracies)}\n")
            file.write(f"Precision: {np.mean(precision_scores)}\n")
            file.write(f"Recall: {np.mean(recall_scores)}\n")
            file.write(f"MSE: {np.mean(mse_scores)}\n\n")

        print("Accuracy : " + str((sum(accuracies) / len(accuracies))))
        print("Precision : " + str((sum(recall_scores) / len(recall_scores))))
        print("Recall : " + str((sum(precision_scores) / len(precision_scores))))
        print("MSE : " + str((sum(mse_scores) / len(mse_scores))))
        print()

        # Dictionary Summary
    print('------------------')
    print('Accuracy : ', cross_validate_result)
    print('Precision : ', cross_validate_precision)
    print('Recall : ', cross_validate_recall)
    print('MSE : ', cross_validate_mse)

    file.write('------------------\n')
    file.write('Summary\n')
    file.write('Accuracy Summary:\n')
    file.write(str(cross_validate_result) + '\n')
    file.write('Precision Summary:\n')
    file.write(str(cross_validate_precision) + '\n')
    file.write('Recall Summary:\n')
    file.write(str(cross_validate_recall) + '\n')
    file.write('MSE Summary:\n')
    file.write(str(cross_validate_mse) + '\n')
