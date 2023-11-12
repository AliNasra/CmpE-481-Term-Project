import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder



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
X_train_t , X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_pre = preprocess(X_train, y_train)    # whole training set or 0.8 of the training set ???

"""
svm_accuracy_val, svm_accuracy_test, svm_test_report = apply_svm(X_train_t, y_train_t, X_val, y_val, X_test, y_test)

print(f"Accuracy for validation set using svm: {svm_accuracy_val}")
print(f"Accuracy for test set using svm: {svm_accuracy_test}")
print(f"Classification report for test set using svm: {svm_test_report}")
"""