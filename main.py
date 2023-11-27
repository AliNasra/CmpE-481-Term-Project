import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def preprocess(X, y):
    ## Preprocessing: Removing autocorrelated features
    redundantFeatures = set()
    redundantFeatures.add("rerun_ID") 
    
    significant_features = {}
    corr = X.corr()
    no_feature = len(corr)
    
    # Remove autocorrelated features
    for i in range(no_feature):
        for j in range(i):
            if abs(corr.iloc[i,j]) > 0.75:
                feature_name = corr.columns[i]
                redundantFeatures.add(feature_name)
    X = X.drop(labels=redundantFeatures, axis=1)

    ## Preprocessing: Feature Selection
    alpha = 0.05
    f_statistic, p_statistic = f_classif(X, y)   # from sklearn.feature_selection
    significant_features["Feature"] = X.columns
    significant_features["P_Statistic"] = p_statistic
    df = pd.DataFrame(significant_features)
    insignificant_features =  df.loc[(df['P_Statistic'] > alpha)].loc[:,"Feature"].tolist()
    X = X.drop(labels = insignificant_features, axis=1)
    
    return X

# hyper parameter tuning for svm
def param_tuning_svm(X_train, y_train):
    X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    X_train_t_pre = preprocess(X_train_t, y_train_t)  # 0.8 of the training set 

    param_grid = {
        'kernel' : ['linear', 'poly', 'rbf'],
        'C':  [0.01, 0.1, 1],
    }
    """
    'gamma': ['scale', 'auto', 1, 0.1],
    'coef0':[0.0, 0.1, 0.2, 0.3],
    'degree':[2, 3, 4, 5]
    """

    clf = SVC()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-2)

    grid_search.fit(X_train_t_pre, y_train_t)
    print("training model done")
    results = grid_search.cv_results_

    print("Best parameters : ", grid_search.best_params_)
    print("Average Score for Validation Set : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores for Validation Set: {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train_t_pre.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    y_pred_val = grid_search.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy for Validation Set: {:.2f}%".format(accuracy_val * 100))  
    val_report = classification_report(y_val, y_pred_val)
    print("Classification report for validation set :")
    print(val_report)

    """
    # randomized search cv
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=3, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)
    print("random search done")
    random_search.fit(X_train_t_pre, y_train_t)
    """

    return grid_search.best_params_

"""
# retrain model with the whole train set and predict on test set
def apply_svm(X_train, X_test, y_train, y_test):
    eliminated_columns = [feature for feature in X_test.columns if feature not in X_train.columns ]
    X_test = X_test.drop(eliminated_columns, axis=1)

    params = param_tuning_svm(X_train, y_train)

    X_train_pre = preprocess(X_train, y_train)  

    clf = SVC(kernel=params['kernel'], C=params['C']) #, gamma=params['gamma'], coef0=params['coef0'], degree=params['degree']) 
    clf.fit(X_train_pre, y_train)
    y_pred_test = clf.predict(X_test)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Accuracy for Test Set: {:.2f}%".format(accuracy_test * 100))  
    #test_report = classification_report(y_test, y_pred_test)
    #print(f"Classification report for test set : {test_report}")
"""

def param_tuning_svm_pca(X_train, y_train):
    X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    X_train_t_pre = preprocess(X_train_t, y_train_t)  # 0.8 of the training set 

    pca = PCA()
    svm = SVC()
    pipe = Pipeline([('pca', pca), ('svm', svm)])

    param_grid = {
        'pca__n_components': [2, 3, 4], 
        'svm__C': [0.01, 0.1, 1], 
        'svm__gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' kernel
        'svm__kernel': ['linear', 'rbf']  # SVM kernel type
    }

    random_search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid,
                                   scoring='accuracy', cv=5, random_state=42, n_iter=10, n_jobs=-2)

    random_search.fit(X_train_t_pre, y_train_t)
    print("training model done")

    best_estimator = random_search.best_estimator_
    results = random_search.cv_results_

    print("Best parameters : ", random_search.best_params_)
    print("Average Score for Validation Set : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores for Validation Set: {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train_t_pre.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    y_pred_val = random_search.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy for Validation Set: {:.2f}%".format(accuracy_val * 100))  
    val_report = classification_report(y_val, y_pred_val)
    print("Classification report for validation set :")
    print(val_report)


def param_tuning_k_nearest_neighbour(X_train, y_train):
    X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    X_train_t_pre = preprocess(X_train_t, y_train_t)  # 0.8 of the training set 

    clf = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [7, 9, 11, 13, 15],
        'p': [1, 2],   
        # p : Minkowski metric                     
        # p = 1: Manhattan distance (L1 norm)  
        # p = 2: Euclidean distance (L2 norm)
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size' : [30, 90, 150, 210],
        'metric' : ["euclidean","manhattan","chebyshev","minkowski"]
    }  
    
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)

    random_search.fit(X_train_t_pre, y_train_t)

    results = random_search.cv_results_

    print("Best parameters : ", random_search.best_params_)
    print("Average Score for Validation Set : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores for Validation Set: {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train_t_pre.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)
  
    y_pred_val = random_search.predict(X_val)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy for Validation Set: {:.2f}%".format(accuracy_val * 100))  
    val_report = classification_report(y_val, y_pred_val)
    print("Classification report for validation set :")
    print(val_report)


def param_tuning_adaboost(X_train, y_train):
    X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    X_train_t_pre = preprocess(X_train_t, y_train_t)  # 0.8 of the training set 

    clf = AdaBoostClassifier()  
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
    }  
    
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)

    random_search.fit(X_train_t_pre, y_train_t)

    results = random_search.cv_results_

    print("Best parameters : ", random_search.best_params_)
    print("Average Score for Validation Set : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores for Validation Set: {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train_t_pre.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)
  
    y_pred_val = random_search.predict(X_val)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy for Validation Set: {:.2f}%".format(accuracy_val * 100))  
    val_report = classification_report(y_val, y_pred_val, zero_division=1)
    print("Classification report for validation set :")
    print(val_report)


data = pd.read_csv('star_classification.csv', index_col=0)
X = data.iloc[:,data.columns != "class"]
y = data.loc[:,"class"]

## to convert categorical labels to numeric labels
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
# 'GALAXY': 0  'QSO': 1  'STAR': 2
# class_weights = {0: 0.59, 1: 0.22, 2: 0.19}

original_features = X.columns.tolist()

## Dividing the data by 0.8*0.8 - 0.8*0.2 - 0.2 ratio (training - validation - testing)
X_train , X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=0)
  
print("K-Nearest Neighbor")
param_tuning_k_nearest_neighbour(X_train, y_train)

#print("SVM")
#param_tuning_svm_pca(X_train, y_train)

print("AdaBoost")
param_tuning_adaboost(X_train, y_train)
