#########################################
# End To End Diabetes Machine Learning Pipeline I
#########################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import joblib
import pandas as pd
import seaborn as sns
from catboost.utils import eval_metric
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skompiler.toskast.sklearn.common import classifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from cart import cart_params
from knn import knn_params
from logistic_regression import cv_results
from salary_predict import num_cols, cat_but_car

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

#########################################
# 1. Exploratory Data Analysis
#########################################

def check_df(dataframe, head=5):
    print("######## shape ##########")
    print(dataframe.shape)
    print("######## types ##########")
    print(dataframe.dtypes)
    print("######## head ##########")
    print(dataframe.head(head))
    print("######## tail ##########")
    print(dataframe.tail(head))
    print("######## NA ##########")
    print(dataframe.isnull().sum())
    print("######## quantiles ##########")
    print(dataframe.describe([0,0.25,0.5,0.75,0.99]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles= [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def grab_col_names(dataframe , cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    --------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols un içerisinde
    Return olan 3 liste toplam değişken sayısına eşittir

    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]
    print(num_but_cat)
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
    print(cat_but_car)
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    print(cat_cols)

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols , num_cols, cat_but_car

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}, end = "\n\n\n"))

def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap="RdBu")
    plt.show()

df = pd.read_csv('datasets/diabetes.csv')

check_df(df)

# değişken türlerinin ayrıştırılması

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col, plot=True)

# sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)

# sayısal değişkenlerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# target ile sayısal değişkenlerin incelenmesi
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#########################################
# 2. Data Preprocessing & Feature Engineering
#########################################

def outlier_thresholds(dataframe , col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range  = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low , up = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up, variable] = up
    dataframe.loc[dataframe[variable] < low, variable] = low

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe

# değişken isimlerini büyütmek
df.columns = [col.upper() for col in df.columns]

#Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x = df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

#Age
df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

#BMI
df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                             labels=["underweight", "healthy", "overweight", "obese"])

# BloodPressure
df['NEW_BLOODPRESSURE'] = pd.cut(x=df['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                 labels=["normal", "hs1", "hs2"])

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "OUTCOME", col)

cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)
check_df(df)

df.columns = [col.upper() for col in df.columns]

# güncel grabcolnames
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

replace_with_thresholds(df, "INSULIN")

#standartlaştırma

X_scaled = StandardScaler().fit_transform(df[num_cols])

df[num_cols] = pd.DataFrame(X_scaled, columns = df[num_cols].columns)

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis = 1)

check_df(df)

def diabetes_data_prep(dataframe):
    def check_df(dataframe, head=5):
        print("######## shape ##########")
        print(dataframe.shape)
        print("######## types ##########")
        print(dataframe.dtypes)
        print("######## head ##########")
        print(dataframe.head(head))
        print("######## tail ##########")
        print(dataframe.tail(head))
        print("######## NA ##########")
        print(dataframe.isnull().sum())
        print("######## quantiles ##########")
        print(dataframe.describe([0, 0.25, 0.5, 0.75, 0.99]).T)

    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist()
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

        Parameters
        ----------
        dataframe: dataframe
            değişken isimleri alınmak istenen dataframedir
        cat_th: int,float
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, float
            kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        -------
        cat_cols: list
            kategorik değişken listesi
        num_cols: list
            numerik değişken listesi
        cat_but_car: list
            kategorik görünümlü kardinal değişken listesi

        Notes
        --------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols un içerisinde
        Return olan 3 liste toplam değişken sayısına eşittir

        """
        cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
        num_cols = [col for col in num_cols if col not in cat_cols]
        num_but_cat = [col for col in dataframe.columns if
                       dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]
        print(num_but_cat)
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
        print(cat_but_car)
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        print(cat_cols)

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cols: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")

        return cat_cols, num_cols, cat_but_car

    def target_summary_with_cat(dataframe, target, categorical_col):
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

    def target_summary_with_num(dataframe, target, numerical_col):
        print(dataframe.groupby(target).agg({numerical_col: "mean"}, end="\n\n\n"))

    def correlation_matrix(dataframe, cols):
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                          cmap="RdBu")
        plt.show()

    dataframe = pd.read_csv('datasets/diabetes.csv')

    check_df(dataframe)

    # değişken türlerinin ayrıştırılması

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    # kategorik değişkenlerin incelenmesi
    for col in cat_cols:
        cat_summary(dataframe, col, plot=True)

    # sayısal değişkenlerin incelenmesi
    dataframe[num_cols].describe().T

    for col in num_cols:
        num_summary(dataframe, col, plot=True)

    # sayısal değişkenlerin birbirleri ile korelasyonu
    correlation_matrix(dataframe, num_cols)

    # target ile sayısal değişkenlerin incelenmesi
    for col in num_cols:
        target_summary_with_num(dataframe, "Outcome", col)

    #########################################
    # 2. Data Preprocessing & Feature Engineering
    #########################################

    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quantile1 = dataframe[col_name].quantile(q1)
        quantile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quantile3 - quantile1
        up_limit = quantile3 + 1.5 * interquantile_range
        low_limit = quantile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
        low, up = outlier_thresholds(dataframe, col_name, q1, q3)
        if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
            return True
        else:
            return False

    def replace_with_thresholds(dataframe, variable):
        low, up = outlier_thresholds(dataframe, variable)
        dataframe.loc[dataframe[variable] > up, variable] = up
        dataframe.loc[dataframe[variable] < low, variable] = low

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    # değişken isimlerini büyütmek
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Glucose
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    # Age
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                 labels=["underweight", "healthy", "overweight", "obese"])

    # BloodPressure
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                     labels=["normal", "hs1", "hs2"])


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    for col in cat_cols:
        cat_summary(dataframe, col)

    for col in cat_cols:
        target_summary_with_cat(dataframe, "OUTCOME", col)

    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    check_df(dataframe)

    dataframe.columns = [col.upper() for col in dataframe.columns]

    # güncel grabcolnames
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    for col in num_cols:
        print(col, check_outlier(dataframe, col, 0.05, 0.95))

    replace_with_thresholds(dataframe, "INSULIN")

    # standartlaştırma

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])

    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    y = dataframe["OUTCOME"]
    X = dataframe.drop("OUTCOME", axis=1)

    return X, y

df = pd.read_csv('datasets/diabetes.csv')
check_df(df)

X, y = diabetes_data_prep(df)

check_df(X)

#########################################
# 3. Base Models
#########################################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models..............")
    classifiers = [('LR', LogisticRegression(max_iter=1000)),
                   ('KNN', KNeighborsClassifier()),
                   ('SVC', SVC()),
                   ('CART', DecisionTreeClassifier()),
                   ('RF', RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier(algorithm='SAMME')),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                  ]
    for name, model in classifiers:
        try:
            cv_results = cross_validate(model, X, y, scoring=scoring, cv=3, error_score='raise')
            print(f"{scoring} : {round(cv_results['test_score'].mean(), 4)} ({name})")
        except Exception as e:
            print(f"{name} failed with error: {e}")

y = (y > 0).astype(int)

base_models(X, y)

#########################################
# 4. Automated Hyperparameter Optimization
#########################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               'min_samples_split': range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                  "n_estimators": [300, 500]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ('CART', DecisionTreeClassifier(), cart_params),
               ('RF', RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization.........")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"###########{name}#############")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models
best_models = hyperparameter_optimization(X, y)

#########################################
# 5. Stacking & Ensemble Learning
#########################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier..........")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])], voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1: {cv_results['test_f1'].mean()}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

#########################################
# 6. Prediction for a New Observation
#########################################

random_user = X.sample(1, random_state=45)

voting_clf.predict(random_user)

joblib.dump(voting_clf, 'voting_clf.pkl')

new_model = joblib.load('voting_clf.pkl')
new_model.predict(random_user)