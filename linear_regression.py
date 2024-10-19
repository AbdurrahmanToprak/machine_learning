import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('datasets/advertising.csv')
df.shape

X = df[["TV"]]
y = df[["sales"]]

#########################
# model
########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv nin katsayısı (w1)
reg_model.coef_[0][0]

###################
# Tahmin
###################

# 150 birimlik tv harcaması olsa ne kadar satış olması beklenir
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

df.describe().T

#Modelin Görselleştirilmesi

g = sns.regplot(x=X, y=y, scatter_kws={"color" : "b", "s": 9},
                ci=False, color="r")

g.set_title(f"Model denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()

#####################
## Tahmin Başarısı
#####################
#MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
#10.51
y.mean()
y.std()

#RMSE
np.sqrt(mean_squared_error(y, y_pred))
#3.24

#MAE
mean_absolute_error(y, y_pred)
#2.54

#R-KARE
reg_model.score(X, y)
#0.61

#################################
# Multiple Linear Regression
#################################

df = pd.read_csv('datasets/advertising.csv')

X = df.drop("sales", axis=1)
y = df["sales"]

#################
# model
#################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

######################
# Tahmin
#####################
yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)


######################
# Tahmin Başarı
#####################

# TRAIN RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN R-KARE
reg_model.score(X_train, y_train)

#TEST RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# TEST R-KARE
reg_model.score(X_train, y_train)

# 10 KATLI CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring='neg_mean_squared_error')))

#################################################
## Simple Linear Regression With Gradient Descent from Scratch
#################################################

# Cost Function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0 , m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# update weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0 , m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

#train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print(f"Iter: {i} b={b:.4f}  w={w:.4f} MSE: {mse:.4f}")
    print(f"After {num_iters} iterations b = {b}, w = {w}, mse = {cost_function(Y, b, w, X)}")
    return cost_history, b, w

df = pd.read_csv('datasets/advertising.csv')
X = df["radio"]
Y = df["sales"]

learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
