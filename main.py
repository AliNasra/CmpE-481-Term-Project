import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,  StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


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
def apply_svm_with_param_tuning(X_train, y_train, X_val, y_val):
    param_grid = {
        'svc__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__C':  [0.1, 1, 10, 50, 100],
        'svc__gamma': ['scale', 'auto'],
        'svc__coef0':[0.0, 0.1, 0.2],
        'svc__degree':[2, 3, 4, 5]
    }

    clf = SVC()
    pipeline = make_pipeline(StandardScaler(), clf)
    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)
    
    random_search.fit(X_train, y_train)
    results = random_search.cv_results_

    print("Best parameters : ", random_search.best_params_)
    print("Average Score : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores: {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    y_pred_val = random_search.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val) * 100
    print("Accuracy for Validation Set: {:.2f}%".format(accuracy_val))  
    return accuracy_val


# retrain model with the whole train set and predict on test set
def apply_svm(X_train, y_train, X_test, y_test, params, eliminate_features):
    X_train = X_train.drop(eliminate_features, axis=1)
    # For test set
    eliminated_columns = [feature for feature in X_test.columns if feature not in X_train.columns ]
    X_test = X_test.drop(eliminated_columns, axis=1)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel=params['svc__kernel'], C=params['svc__C'], gamma=params['svc__gamma'], 
              coef0=params['svc__coef0'], degree=params['svc__degree']) 
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    accuracy_test = accuracy_score(y_test, y_pred_test) * 100
    print("SVM Accuracy for Test Set: {:.2f}%".format(accuracy_test))  



def apply_knn_with_param_tuning(X_train, y_train, X_val, y_val):
    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5], 
        'p': [1, 2],   
        # p : Minkowski metric                     
        # p = 1: Manhattan distance (L1 norm)  
        # p = 2: Euclidean distance (L2 norm)
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size' : [30, 90, 150, 210],
        'metric' : ["euclidean","manhattan","chebyshev","minkowski"]
    }  
    
    clf = KNeighborsClassifier()
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)
    random_search.fit(X_train, y_train)

    results = random_search.cv_results_

    print("Best parameters : ", random_search.best_params_)
    print("Average Score : {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Scores : {:.2f}".format(np.std(results['mean_test_score'])))

    # For validation set
    y_pred_val = random_search.predict(X_val)

    accuracy_val = accuracy_score(y_val, y_pred_val) * 100
    print("Accuracy Score for Validation Set: {:.2f}%".format(accuracy_val))  
    return accuracy_val


def apply_knn(X_train, y_train, X_test, y_test, params, eliminate_features):
    X_train = X_train.drop(eliminate_features, axis=1)
    # For test set
    eliminated_columns = [feature for feature in X_test.columns if feature not in X_train.columns ]
    X_test = X_test.drop(eliminated_columns, axis=1)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'], p=params['p'], weights=params[ 'weights'], 
              algorithm=params['algorithm'], leaf_size=params['leaf_size'], metric=params['metric']) 
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    accuracy_test = accuracy_score(y_test, y_pred_test) * 100
    print("KNN Accuracy for Test Set: {:.2f}%".format(accuracy_test))  


def apply_randomForest_with_param_tuning(X_train, y_train, X_val, y_val):
    clf = RandomForestClassifier()
    param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
    }
    # Use RandomizedSearchCV to search for the best parameters and perform k-fold cross-validation
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-2)
    random_search.fit(X_train, y_train)

    """
    # printing feature importance graph
    best_rf_estimator = random_search.best_estimator_
    feature_importances = best_rf_estimator.feature_importances_

    f, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feature_importances)), feature_importances, color='midnightblue', alpha=.7)
    ax.set_yticks(range(len(feature_importances)))
    ax.set_yticklabels(X_train.columns)
    ax.set_title("Importance of each feature")
    ax.set_xlabel("Importance")
    plt.show()
    """

    print("Best parameters used in Random Forest: ", random_search.best_params_)
    results = random_search.cv_results_
    print("Average Test Score: {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Test Scores: {:.2f}".format(np.std(results['mean_test_score'])))

    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    y_pred = random_search.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred) * 100
    print("Accuracy Score for Validation Set: {:.2f}%".format(accuracy))  
    return accuracy


def apply_randomForest(X_train, y_train, X_test, y_test, params, eliminate_features):
    X_train = X_train.drop(eliminate_features, axis=1)
    # For test set
    eliminated_columns = [feature for feature in X_test.columns if feature not in X_train.columns ]
    X_test = X_test.drop(eliminated_columns, axis=1)
 
    clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                 min_samples_split=params[ 'min_samples_split'], min_samples_leaf=params['min_samples_leaf']) 
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test) * 100
    print("Random Forest Accuracy for Test Set: {:.2f}%".format(accuracy_test)) 


def apply_XGBoost_with_param_tuning(X_train, y_train, X_val, y_val):
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42)
    param_grid = {
        'max_depth': [1, 3, 6, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [20, 50, 100, 150, 200],
    }
    # Use RandomizedSearchCV to search for the best parameters and perform k-fold cross-validation
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    print("Best parameters used in XGBoost: ", random_search.best_params_)
    results = random_search.cv_results_
    print("Average Test Score: {:.2f}".format(np.mean(results['mean_test_score'])))
    print("Standard Deviation of Test Scores: {:.2f}".format(np.std(results['mean_test_score'])))

    eliminated_columns = [feature for feature in X_val.columns if feature not in X_train.columns ]
    X_val = X_val.drop(eliminated_columns, axis=1)

    y_pred = random_search.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred) * 100
    print("Accuracy Score for Validation Set: {:.2f}%".format(accuracy))  
    return accuracy


def apply_XGBoost(X_train, y_train, X_test, y_test, params, eliminate_features):
    X_train = X_train.drop(eliminate_features, axis=1)
    # For test set
    eliminated_columns = [feature for feature in X_test.columns if feature not in X_train.columns ]
    X_test = X_test.drop(eliminated_columns, axis=1)

    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42, max_depth=params['max_depth'], 
                            learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test) * 100
    print("XGBoost Accuracy for Test Set: {:.2f}%".format(accuracy_test)) 


def accuracy_graph(feature_counts, val_accuracy_values, color):
    plt.figure(figsize=(8, 6))
    plt.plot(feature_counts, val_accuracy_values, marker='o', color=color, alpha=1, linestyle='-')
    plt.title('Accuracy vs. Feature Count')
    plt.xlabel('Number of Features')
    plt.xticks(feature_counts)
    plt.ylabel('Accuracy')
    plt.grid(True)

    for i, (x, y) in enumerate(zip(feature_counts, val_accuracy_values)):
        plt.text(x, y, f'{y:.2f}', ha='right' if i % 2 == 0 else 'left', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


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
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=0)
X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
X_train_t_pre = preprocess(X_train_t, y_train_t)  # decreases features from 17 to 10
feature_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17]
important_features = ['cam_col', 'field_ID', 'fiber_ID', 'run_ID', 'alpha', 'delta', 'spec_obj_ID', 'r', 'u', 'redshift'] # from least to most


print("SVM")
svm_val_accuracy_values = []
X_train_svm= X_train_t_pre
# feature_count == 17 without preprocessing
print("feature_count: 17")
val_accuracy = apply_svm_with_param_tuning(X_train_t, y_train_t, X_val, y_val)
svm_val_accuracy_values.insert(0,val_accuracy)
# feature_count == 10 after preprocessing
print("feature_count: 10")
val_accuracy= apply_svm_with_param_tuning(X_train_t_pre, y_train_t, X_val, y_val)
svm_val_accuracy_values.insert(0,val_accuracy)
for i in range(len(feature_counts)-2):
    #drop the least important feature 
    print("feature count= ", feature_counts[len(feature_counts)-3-i])
    X_train_svm = X_train_svm.drop([important_features[i]], axis=1)
    val_accuracy= apply_svm_with_param_tuning(X_train_svm, y_train_t, X_val, y_val)
    svm_val_accuracy_values.insert(0,val_accuracy)

accuracy_graph(feature_counts, svm_val_accuracy_values, 'blue')
print()
# best feature count and best parameters are selected manually after evaluation of graph above
best_feature_count = 6
best_params = {'svc__kernel': 'rbf', 'svc__gamma': 'auto', 'svc__degree': 5, 'svc__coef0': 0.0, 'svc__C': 100}
apply_svm(X_train_t_pre, y_train_t, X_test, y_test, best_params, important_features[0:10-best_feature_count])
print("\n\n")


print("K-Nearest Neighbor Without Oversampling") 
knn_val_accuracy_values = []
X_train_knn= X_train_t_pre
# feature_count == 17 without preprocessing
print("feature_count: 17")
val_accuracy = apply_knn_with_param_tuning(X_train_t, y_train_t, X_val, y_val)
knn_val_accuracy_values.insert(0,val_accuracy)
# feature_count == 10 after preprocessing
print("feature_count: 10")
val_accuracy= apply_knn_with_param_tuning(X_train_t_pre, y_train_t, X_val, y_val)
knn_val_accuracy_values.insert(0,val_accuracy)
for i in range(len(feature_counts)-2):
    #drop the least important feature 
    print("feature_count: ", feature_counts[len(feature_counts)-3-i])
    X_train_knn = X_train_knn.drop([important_features[i]], axis=1)
    val_accuracy= apply_knn_with_param_tuning(X_train_knn, y_train_t, X_val, y_val)
    knn_val_accuracy_values.insert(0,val_accuracy)   

accuracy_graph(feature_counts, knn_val_accuracy_values, 'green')
print()
# best feature count and best parameters are selected manually after evaluation of graph above
best_feature_count = 2
best_params = {'weights': 'uniform', 'p': 2, 'n_neighbors': 5, 'metric': 'manhattan', 'leaf_size': 210, 'algorithm': 'brute'}
apply_knn(X_train_t_pre, y_train_t, X_test, y_test, best_params, important_features[0:10-best_feature_count])
print("\n\n")

"""
# Printing class distribution graph
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(['Galaxy','QSO','STAR'], counts, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('KNN - Number of Instances in Each Class')
plt.show()
"""

print("K-nearest Neighbor with oversampling")
sm = SMOTE(random_state=42, k_neighbors = 5)          # Oversampling
X_train_t_sam, y_train_t_sam = sm.fit_resample(X_train_t, y_train_t)
X_train_t_sam_pre = preprocess(X_train_t_sam, y_train_t_sam)
knn_val_accuracy_values = []
X_train_knn= X_train_t_sam_pre 
# feature_count == 17 without preprocessing
print("feature count: 17")
val_accuracy= apply_knn_with_param_tuning(X_train_t_sam, y_train_t_sam, X_val, y_val)
knn_val_accuracy_values.insert(0,val_accuracy)
# feature_count == 10 after preprocessing
print("feature count: 10")
val_accuracy= apply_knn_with_param_tuning(X_train_t_sam_pre, y_train_t_sam, X_val, y_val)
knn_val_accuracy_values.insert(0,val_accuracy)
for i in range(len(feature_counts)-2):
    #drop the least important feature 
    print("feature count: ", feature_counts[len(feature_counts)-3-i])
    X_train_knn = X_train_knn.drop([important_features[i]], axis=1)
    val_accuracy= apply_knn_with_param_tuning(X_train_knn, y_train_t_sam, X_val, y_val)
    knn_val_accuracy_values.insert(0,val_accuracy)

accuracy_graph(feature_counts, knn_val_accuracy_values, 'red')
# best feature count and best parameters are selected manually after evaluation of graph above
best_feature_count = 2
best_params = {'weights': 'uniform', 'p': 2, 'n_neighbors': 5, 'metric': 'manhattan', 'leaf_size': 210, 'algorithm': 'brute'}
apply_knn(X_train_t_sam_pre, y_train_t_sam, X_test, y_test, best_params, important_features[0:10-best_feature_count])
print("\n\n")


print("Random Forest")
rf_val_accuracy_values = []
X_train_rf= X_train_t_pre
# feature_count == 17 without preprocessing
print("feature_count: 17")
val_accuracy = apply_randomForest_with_param_tuning(X_train_t, y_train_t, X_val, y_val)
rf_val_accuracy_values.insert(0,val_accuracy)
# feature_count == 10 after preprocessing
print("feature_count: 10")
val_accuracy= apply_randomForest_with_param_tuning(X_train_t_pre, y_train_t, X_val, y_val)
rf_val_accuracy_values.insert(0,val_accuracy)
for i in range(len(feature_counts)-2):
    #drop the least important feature 
    print("feature count= ", feature_counts[len(feature_counts)-3-i])
    X_train_rf = X_train_rf.drop([important_features[i]], axis=1)
    val_accuracy= apply_randomForest_with_param_tuning(X_train_rf, y_train_t, X_val, y_val)
    rf_val_accuracy_values.insert(0,val_accuracy)

accuracy_graph(feature_counts, rf_val_accuracy_values, 'blue')
print()
# best feature count and best parameters are selected manually after evaluation of graph above
best_feature_count = 10
best_params = {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}
apply_randomForest(X_train_t_pre, y_train_t, X_test, y_test, best_params, important_features[0:10-best_feature_count])
print("\n\n")


print("XGBoost")
xgb_val_accuracy_values = []
X_train_xgb= X_train_t_pre
# feature_count == 17 without preprocessing
print("feature_count: 17")
val_accuracy = apply_XGBoost_with_param_tuning(X_train_t, y_train_t, X_val, y_val)
xgb_val_accuracy_values.insert(0,val_accuracy)
# feature_count == 10 after preprocessing
print("feature_count: 10")
val_accuracy= apply_XGBoost_with_param_tuning(X_train_t_pre, y_train_t, X_val, y_val)
xgb_val_accuracy_values.insert(0,val_accuracy)
for i in range(len(feature_counts)-2):
    #drop the least important feature 
    print("feature count= ", feature_counts[len(feature_counts)-3-i])
    X_train_xgb = X_train_xgb.drop([important_features[i]], axis=1)
    val_accuracy= apply_XGBoost_with_param_tuning(X_train_xgb, y_train_t, X_val, y_val)
    xgb_val_accuracy_values.insert(0,val_accuracy)

accuracy_graph(feature_counts, xgb_val_accuracy_values, 'green')
print()
# best feature count and best parameters are selected manually after evaluation of graph above
best_feature_count = 10
best_params = {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1}
apply_XGBoost(X_train_t_pre, y_train_t, X_test, y_test, best_params, important_features[0:10-best_feature_count])
print("\n\n")

