import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_val_score



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


def apply_svm(X_train_t, y_train_t, X_val, y_val, X_test, y_test):
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_train_t, y_train_t)

    # For validation set
    y_pred_val = clf.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)

    # For test set
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    test_report = classification_report(y_test, y_pred_test)

    return accuracy_val, accuracy_test, test_report

def apply_randomForest(X_train, y_train, kf):
    X_pre = preprocess(X_train, y_train)    # whole training set or 0.8 of the training set
    #X_train_t , X_val, y_train_t, y_val = train_test_split(X_pre, y_train, test_size=0.2, random_state=0)
    accuracy_scores = {}
    
    for max_depth in range(2,10,1):
        for n_estimators in range (10,200,10):
            for min_samples_leaf in range(1,10,1):
                clf = RandomForestClassifier(max_depth=max_depth,n_estimators = n_estimators,min_samples_leaf = min_samples_leaf, random_state=0)  
                scores = cross_val_score(clf, X_pre, y_train, cv = kf)
                accuracy_scores[clf] = scores.mean()
                print("Max_depth is ", max_depth, "n_estimators is ", n_estimators," min_samples_leaf is  ",min_samples_leaf)
                print("Score is ",scores.mean())
        #test_report = classification_report(y_test, y_pred_test)
    #return accuracy_list
    #return accuracy_val, accuracy_test, test_report
    


data = pd.read_csv('star_classification.csv', index_col=0)
X = data.iloc[:,data.columns != "class"]
y = data.loc[:,"class"]

## to convert categorical labels to numeric labels
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)
# 'GALAXY': 0  'QSO': 1  'STAR': 2

original_features = X.columns.tolist()

## Dividing the data by 0.8*0.8 - 0.8*0.2 - 0.2 ratio (training - validation - testing)
X_train , X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=0)
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
apply_randomForest(X_train,y_train,kf)

"""
svm_accuracy_val, svm_accuracy_test, svm_test_report = apply_svm(X_train_t, y_train_t, X_val, y_val, X_test, y_test)

print(f"Accuracy for validation set using svm: {svm_accuracy_val}")
print(f"Accuracy for test set using svm: {svm_accuracy_test}")
print(f"Classification report for test set using svm: {svm_test_report}")
"""