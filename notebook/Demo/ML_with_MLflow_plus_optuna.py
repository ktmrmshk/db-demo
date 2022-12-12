# Databricks notebook source
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# COMMAND ----------

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
display(X)
display(y)

# COMMAND ----------

# MAGIC %md ### 通常のモデル学習 - 1回のみ実行

# COMMAND ----------

reg = RandomForestRegressor(
    n_estimators=25,
    bootstrap=True,
    max_depth=10
)
## 通常の学習
#reg.fit(X, y)
#reg.score(X, y)

## Cross Validation
r2_score = cross_val_score(reg, X, y, cv=3).mean()
print(r2_score)

# COMMAND ----------

# MAGIC %md ### HyperOptを使って、ハイパーパラメータチューニングを並列化させる

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, rand

# COMMAND ----------

# paramsを受け取って、最小値化させたい値を返す関数を作る
# 今回の場合、以下の通り
## params: 機械学習のハイパーパラメータ
## 最小化の値: -(学習結果のR2値) ## マイナスをつける理由は、R2は大きいほど良いので、最小化としてはマイナスが必要。


def objective(params):
    reg = RandomForestRegressor(
        n_estimators = int( params['n_estimators'] ),
        bootstrap    = params['bootstrap'],
        max_depth    = int( params['max_depth'])
    )
    
    r2_score = cross_val_score(reg, X, y, cv=3).mean()
    
    return {'status': STATUS_OK, 'loss': -r2_score}

# paramsの範囲を与える
search_space = {
    'n_estimators': hp.quniform('n_estimators', 1, 50, 1),
    'bootstrap':    hp.choice('bootstrap', [True, False]),
    'max_depth':    hp.quniform('max_depth', 1, 20, 1),
}


# COMMAND ----------

import mlflow

#spark_trials = SparkTrials(parallelism=8)
spark_trials = SparkTrials()

with mlflow.start_run():
    best_result = fmin(
        fn        = objective,
        space     = search_space,
        algo      = tpe.suggest,
        max_evals = 30, # 探索のトライ数の上限
        trials    = spark_trials)

# COMMAND ----------

import hyperopt

# bestスコアだったパラメータを確認
best_params = hyperopt.space_eval( search_space, best_result )
print(best_params)

# COMMAND ----------

# bestスコアだったパラメータで再学習(Mlflowでモデルトラックさせるため。Hyperopt中はパラメータだけのトラックになるため)

best_reg = reg = RandomForestRegressor(
        n_estimators = int( best_params['n_estimators'] ),
        bootstrap    = best_params['bootstrap'],
        max_depth    = int( best_params['max_depth'])
    )

best_reg.fit(X,y)



# COMMAND ----------

# MAGIC %md ### 比較としてOPTUNAで同じことを実施する

# COMMAND ----------

# MAGIC %pip install optuna

# COMMAND ----------

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# COMMAND ----------

import optuna

def objective_optuna(trial):
    bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
    n_estimators =  trial.suggest_int('n_estimators', 1, 50)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    
    reg = RandomForestRegressor(
        n_estimators = n_estimators,
        bootstrap    = bootstrap,
        max_depth    = max_depth
    )
    
    r2_score = cross_val_score(reg, X, y, cv=3).mean()
    
    return r2_score

    

# COMMAND ----------

study = optuna.create_study(direction='maximize')
study.optimize(objective_optuna, n_trials=30)

# COMMAND ----------

print(study.best_params)

# COMMAND ----------


