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
    n_estimators =  trial.suggest_int('n_estimators', 1, 15)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    
    reg = RandomForestRegressor(
        n_estimators = n_estimators,
        bootstrap    = bootstrap,
        max_depth    = max_depth
    )
    
    r2_score = cross_val_score(reg, X, y, cv=3).mean()
    
    return r2_score

    

# COMMAND ----------

study = optuna.create_study(direction='maximize')
study.optimize(objective_optuna, n_trials=5)

# COMMAND ----------

print(study.best_params)

# COMMAND ----------

# MAGIC %md ## 複数モデル作成の並列化(ハイパーパラメーターチューニングではなく、店舗ごとのモデル作成を並列化)
# MAGIC 
# MAGIC 一店舗あたり一つの学習モデルを作成するシナリオで、10店舗あった場合、店舗のモデルを並列的に作成する例。
# MAGIC 
# MAGIC 方針は以下の通り。
# MAGIC 
# MAGIC 1. 全店舗のデータを一つのデルタテーブルにまとめる
# MAGIC 2. 一店舗分のデータを入力とする機械学習モデル作成の処理を関数(UDF化)にする。
# MAGIC 3. 上位のデルタテーブルに対して、Group BYでUDFを適用し、結果を得る

# COMMAND ----------

# MAGIC %md #### 1. 全店舗のデータを一つのデルタテーブルにする。
# MAGIC 
# MAGIC 10店舗分のデータを作る。
# MAGIC 元データは、これまで使ってきたCalforinia_housingで、これに10店舗分の店舗ID(store_id)をクロスjoinさせる。(同じデータが10店舗分できる。)

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X['target'] = y

spark_df_X = spark.createDataFrame(X)
display(spark_df_X)

# COMMAND ----------

# 店舗IDのカラムを作る(10店舗分)
stores = [ (i,) for i in range(10)]
spark_df_store_id = spark.createDataFrame(data = stores, schema=['store_id'])
display(spark_df_store_id)

# COMMAND ----------

#  店舗IDでCrossJoinさせて、10店舗分にCalfornia_housingのデータを複製する(レコード数が10倍になる)
df_cross_joined = spark_df_store_id.crossJoin(spark_df_X)
display(df_cross_joined.sort('AveRooms', 'store_id'))
df_cross_joined.count()

# COMMAND ----------

# メモリに乗り切っているので、このままデータフレームを使えば良い。
# ただし、本番のシナリオは元データがデルタテーブルになっている前提なので、ここでも一旦デルタテーブルにしておく

df_cross_joined.write.mode('overwrite').saveAsTable('calfornia_housing')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- デルタテーブルから読み込みでくるか確認
# MAGIC SELECT * FROM calfornia_housing

# COMMAND ----------

# MAGIC %md #### 2. 一店舗分のデータを入力とする機械学習モデル作成の処理を関数(UDF化)にする。

# COMMAND ----------

def make_model(key, pdf):
  import numpy as np
  import pandas as pd
  import json
  from sklearn.model_selection import cross_val_score
  from sklearn.ensemble import RandomForestRegressor
  import optuna

  X = pdf.drop(['store_id', 'target'], axis=1)
  y = pdf['target']
  
  def objective_optuna(trial):
      bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
      n_estimators =  trial.suggest_int('n_estimators', 1, 15)
      max_depth = trial.suggest_int('max_depth', 1, 10)

      reg = RandomForestRegressor(
          n_estimators = n_estimators,
          bootstrap    = bootstrap,
          max_depth    = max_depth
      )

      r2_score = cross_val_score(reg, X, y, cv=3).mean()
      return r2_score
    
  study = optuna.create_study(direction='maximize')
  study.optimize(objective_optuna, n_trials=3)
  
  ret = []
  ret += key # store_id 
  ret.append( json.dumps(study.best_params) ) # best params as json

#### ベストモデルを再学習もここで実行可能(コメントアウトを削除するとMLflowでトラックされる) 
#   import mlflow
#   mlflow.autolog()
#   with mlflow.start_run():
#     best_reg = RandomForestRegressor(
#       n_estimators=study.best_params['n_estimators'],
#       bootstrap=study.best_params['bootstrap'],
#       max_depth=study.best_params['max_depth']
#     )
#     mlflow.log_param('store_id', key[0] )
#     best_reg.fit(X, y)
  
  return pd.DataFrame([ret])

# COMMAND ----------

####### 上の関数をテスト
# key = (8,)
# pdf = spark.read.table('calfornia_housing').where('store_id = 8').toPandas()
# display(pdf)

# ret_df = make_model(key, pdf)
# display(ret_df)

# COMMAND ----------

# MAGIC %md ### 3. 上位のデルタテーブルに対して、Group BYでUDFを適用し、結果を得る
# MAGIC  

# COMMAND ----------

ret_df = (
  spark.read.table('calfornia_housing')
  .repartition(16)
  .groupBy('store_id')
  .applyInPandas( make_model, schema='store_id int, best_params string')
)

# COMMAND ----------

display(ret_df)
