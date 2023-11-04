import pandas as pd
import sklearn


## Preprocessing: Removing autocorrelated features
data = pd.read_csv('star_classification.csv', index_col=0)
X = data.iloc[:,data.columns != "class"]
y = data.loc[:,"class"]
corr = X.corr()
no_feature = len(corr)
redundantFeatures = set()
for i in range(no_feature):
    for j in range(i):
        if abs(corr.iloc[i,j]) > 0.75:
            feature_name = corr.columns[i]
            redundantFeatures.add(feature_name)
redundantFeatures.add("rerun_ID") # It's a constant
X.drop(labels=redundantFeatures, axis=1, inplace = True)
## Preprocessing: Feature Selection
significant_features = {}
alpha = 0.05
f_statistic, p_statistic = sklearn.feature_selection.f_classif(X, y)
significant_features["Feature"] = X.columns
significant_features["P_Statistic"] = p_statistic
df = pd.DataFrame(significant_features)
insignificant_features =  df.loc[(df['P_Statistic'] > alpha)].loc[:,"Feature"].tolist()
X.drop(labels = insignificant_features, axis=1, inplace=True)
#print(X.columns.tolist())
## Dividing the data by 0.8-0.2 ratio (training - testing)
#X_train , Y_train, X_test, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

