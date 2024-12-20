import warnings
from tabnanny import verbose
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate,validation_curve
from skompiler import skompile

from logistic_regression import X_train, y_train, y_test, cv_results

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv('datasets/diabetes.csv')
##### Data Analysis
##### Data Preprocessing feature extraction

#####Model with CART

y = df['Outcome']
X = df.drop('Outcome', axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

#confusion matrix için y_pred
y_pred = cart_model.predict(X)

#auc için y_prob
y_prob = cart_model.predict_proba(X)[:, 1]

#confusion matrix
print(classification_report(y, y_pred))

#auc
roc_auc_score(y, y_prob)

#####################################
# Holdout yöntemi ile başarı değerlendirme
#####################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

cart_model = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)

#train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

#test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#CV ile başarı değerlendirme

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X,
                            y,
                            scoring=['accuracy','f1','roc_auc'],
                            cv=10)
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

########################################
# Hyperparemeter optimization with GridSearchCv
########################################

cart_model.get_params()

cart_params = {'max_depth' : range(1, 11),
               'min_samples_split' : range(2, 20)}
cart_gs_best = GridSearchCV(cart_model,
                            cart_params,
                            #scoring="roc_auc",
                            cv = 5,
                            n_jobs = -1,
                            verbose = True).fit(X, y)

cart_gs_best.best_params_

cart_gs_best.best_score_

random = X.sample(1, random_state=0)
cart_gs_best.predict(random)
################################
# Final Model
##############################

cart_final = DecisionTreeClassifier(**cart_gs_best.best_params_, random_state=17).fit(X, y)

cart_final = cart_model.set_params(**cart_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(cart_final,
                            X,y,
                            cv = 5,
                            scoring=['accuracy','f1','roc_auc'],)

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################
# Feature Importance
##############################

cart_final.feature_importances_

def plot_feature_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value' : model.feature_importances_, 'Feature' : features.columns})
    plt.figure(figsize = (10,10))
    sns.set(font_scale=1)
    sns.barplot(x = 'Value', y = 'Feature', data = feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('feature_importance.png')

plot_feature_importance(cart_final, X)

################################
# Analyzing Model Complexity with Leaning Curves
##############################

train_score, test_score = validation_curve(cart_final,
                                           X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring='roc_auc',
                                           cv=10)
mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

plt.plot(range(1, 11),
         mean_train_score,
         label = 'train',
         color='blue')
plt.plot(range(1, 11),
         mean_test_score,
         label = 'test',
         color='g')

plt.title('Validation Curve for CART')
plt.xlabel('max_depth')
plt.ylabel('roc_auc')
plt.tight_layout()
plt.legend(loc='best')
plt.show()


def val_curve_params(model, X, y, param_name, param_range, scoring = 'roc_auc', cv = 10):
    train_score, test_score = validation_curve(
        model,X, y, param_name = param_name, param_range = param_range, scoring = scoring, cv = cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label = 'train', color='blue')
    plt.plot(param_range, mean_test_score, label = 'test', color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.legend(loc='best')
    plt.show()

val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring = 'f1')

cart_val_params = [["max_depth" , range(1, 11)] , ["min_samples_split" , range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_final, X, y, cart_val_params[i][0], cart_val_params[i][1])


################################
# Visualization the Desicion Tree
##############################

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(cart_final, col_names = X.columns, file_name = 'cart.png')

################################
# Extraction Decision Rules
##############################

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

##################################
# Extraction Python Codes of Decision Rules
#################################

print(skompile(cart_final.predict).to('python/code'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))

##################################
# Prediction using with python codes
##################################

def predict_with_rules(x):
    return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <=
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

##################################
# Savin and Loading Model
##################################

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)