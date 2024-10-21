import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, StandardScaler

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, \
    confusion_matrix, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv("datasets/hitters.csv")
df.head()

def grab_col_names(dataframe, cat_th = 10, car_th = 20):

    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in dataframe.columns if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[0]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

######################################
## Outliers (Aykırı Değerler)
########################################
def outlier_thresholds(dataframe , col_name, q1=0.10, q3=0.90):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range  = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low , up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[dataframe[variable] > up, variable] = up
    dataframe.loc[dataframe[variable] < low, variable] = low

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

######################################
## Missing Values (Eksik Değerler)
########################################
def missing_values(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')
    if na_name:
        return na_cols

missing_values(df)
df.isnull().sum()

#eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi

missing_values(df, True)
na_cols  = missing_values(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(),
                            'Count' : temp_df.groupby(col)[target].count()}), end='\n\n')

missing_vs_target(df, "Salary", na_cols)



cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe
cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique()>= 2]
df = one_hot_encoder(df, ohe_cols)

### kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#değişkenlerin standartlaştırılması
scaler = RobustScaler()
df= pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()

#knn uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()

#standartlaştırmayı geri alma
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

print(df)

##################################################
#Model & Prediction
################################################

y = df["Salary"]

X = df.drop(["Salary"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#######################################
# Model Evaluation
######################################

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

from sklearn.model_selection import cross_val_score

# Model tanımı
rf_model = RandomForestRegressor(random_state=17)

# K-katlamalı çapraz doğrulama ile performans değerlendirmesi
cv_scores = cross_val_score(rf_model, X, y, cv=5)  # 5 katlı
print(f'Cross-validated scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')


####################################
# Prediction for A New Observation
#######################################

random_user = X.sample(1, random_state=17)
model.predict(random_user)