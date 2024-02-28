import pandas as pd
import seaborn as sns
import optuna

from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('IoTID20_preprocessada.csv')
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

labels = data['Label']
data = data.drop('Label', axis = 1)
feature_list = list(data.columns)

X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size = 0.20,random_state = 42)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 32)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
#    criterion = trial.suggest_categorical('criterion', ["squared_error", "absolute_error", "friedman_mse", "poisson"])
    criterion = trial.suggest_categorical('criterion', ["gini", "log_loss", "entropy"])    

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state= 21
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metric  to optimize
#    score = mean_squared_error(y_test, y_pred)
    score = round(accuracy_score(y_pred,y_test),3)
    
    return score

def modelresults(predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    acc = accuracy_score(y_test,predictions)
    
    print('Mean absolute error on model is {:.4f}'.format(mae))
    print('')
    print('Mean squared error on model is {:.4f}'.format(mse))
    print('')
    print('The r2 score on model is {:.4f}'.format(r2))
    print('')
    print('The accuracy score on model is {:.4f}'.format(acc))

rfr = RandomForestClassifier(random_state = 21)
rfr.fit(X_train, y_train)

y_pred_rfr_fit = rfr.predict(X_test)

modelresults(y_pred_rfr_fit)

#study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
study = optuna.create_study(direction='maximize', study_name='floresta randomica', sampler=optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=100,n_jobs=5)

# Print the best parameters found 
print("Best trial:")
trial = study.best_trial

print("Value: {:.4f}".format(trial.value))

print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params
