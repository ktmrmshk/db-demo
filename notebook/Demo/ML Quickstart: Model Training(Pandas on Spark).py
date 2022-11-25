# Databricks notebook source
# MAGIC %md Databricks ML クイックスタート モデルトレーニング
# MAGIC 
# MAGIC このノートブックでは、Databricksでの機械学習モデルトレーニングの概要を説明します。モデルのトレーニングには、Databricks Runtime for Machine Learningにプリインストールされているscikit-learnなどのライブラリが利用できます。さらに、学習したモデルを追跡するためにMLflowを、ハイパーパラメータのチューニングをスケールするためにSparkTrialsとHyperoptを使用することができます。
# MAGIC 
# MAGIC このチュートリアルでは、以下を取り上げます。
# MAGIC - パート1：MLflowトラッキングを用いたシンプルな分類モデルのトレーニング
# MAGIC - パート2：Hyperoptを使用して、よりパフォーマンスの高いモデルのハイパーパラメータチューニングを行う。
# MAGIC 
# MAGIC モデルライフサイクル管理やモデル推論など、Databricks上での機械学習のプロダクション化の詳細については、ML End to End Example ([AWS](https://docs.databricks.com/applications/mlflow/end-to-end-example.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/end-to-end-example)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/end-to-end-example.html)) を参照してください。
# MAGIC 
# MAGIC ### 要件
# MAGIC - Databricks Runtime 7.5 ML以上が稼働しているクラスタ

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ライブラリ
# MAGIC 必要なライブラリをインポートします。これらのライブラリは、Databricks Runtime for Machine Learning ([AWS](https://docs.databricks.com/runtime/mlruntime.html)|[Azure](https://docs.microsoft.com/azure/databricks/runtime/mlruntime)|[GCP](https://docs.gcp.databricks.com/runtime/mlruntime.html)) クラスタにプリインストールされており、互換性とパフォーマンスのためにチューニングが施されています。

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md 
# MAGIC ### データのロード・前処理
# MAGIC このチュートリアルでは、様々なワインのサンプルを記述したデータセットを使用します。データセット](https://archive.ics.uci.edu/ml/datasets/Wine)はUCI Machine Learning Repositoryからで、DBFSに含まれています([AWS](https://docs.databricks.com/data/databricks-file-system.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/databricks-file-system)|[GCP](https://docs.gcp.databricks.com/data/databricks-file-system.html)).
# MAGIC 赤ワインと白ワインを品質で分類することが目的です。
# MAGIC 
# MAGIC 他のデータソースからのアップロードや読み込みの詳細については、データとの連携に関するドキュメント([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/index)|[GCP](https://docs.gcp.databricks.com/data/index.html))を参照してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法1: 通常のPandasでCSVをロード、前処理をして、最後、DeltaLakeに保存する
# MAGIC 
# MAGIC 読み込むCSVなどが小さいサイズであれば、この方法が早い。通常のPandasなので、前処理自体は分散化されない。
# MAGIC 逆に、読み込むCSVファイルが膨大になる場合は、この後に説明する"Pandas on Spark"を使う方が処理がSpark分散化され、高速になる。
# MAGIC 
# MAGIC とちらの場合にも、最後はDeltaLake形式で保存(永続化)するので、データ結果は変わらない。

# COMMAND ----------

# Load and preprocess data
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')

# 前処理
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# DeltaLakeはカラム名に空白文字を許さないので、ここで修正。
data_df = data_df.rename(columns= lambda s: s.replace(' ', '_') )

# Sparkデータフレームにした後、delta形式で保存
data_df_spark = spark.createDataFrame(data_df)
data_df_spark.write.mode('overwrite').save('/tmp/example_codes/wine_data.delta')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方法2: Pandas on SparkでCSVをロード、前処理をして、最後、DeltaLakeに保存する
# MAGIC 
# MAGIC 方法1の処理をPandas 

# COMMAND ----------

# Pandas on Sparkのimport
import pyspark.pandas as ps

# Pandas on SparkはSparkなので、ファイルパスがSparkと同様にHDFS/オブジェクトストレージパスになる。S3上の場合は、`s3://xxxx`のようなパスにする
white_wine = ps.read_csv("/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine   = ps.read_csv("/databricks-datasets/wine-quality/winequality-red.csv", sep=';')

# 前処理
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = ps.concat([white_wine, red_wine], axis=0)


# DeltaLakeはカラム名に空白文字を許さないので、ここで修正。その後、deltaに書き込む
data_df = data_df.rename(columns= lambda s: s.replace(' ', '_') )
data_df.to_delta('/tmp/example_codes/wine_data.delta', 'w')

# COMMAND ----------

# MAGIC %md ### 前処理した結果の最適化とデータの確認 (これ以降は方法1, 2で同じ)

# COMMAND ----------

# Delta Lakeファイルの最適化(オプショナルだが、実行した方が後続の処理が高速になる)
spark.sql('OPTIMIZE delta.`/tmp/example_codes/wine_data.delta`')

# COMMAND ----------

# Delta Lakeからデータをロード(pandasデータフレーム)
data_df_delta = spark.read.load('/tmp/example_codes/wine_data.delta').toPandas()
data_df_delta.head()

# COMMAND ----------

# Split 80/20 train-test

data_labels = data_df_delta['quality'] >= 7
data_df_delta = data_df_delta.drop(['quality'], axis=1)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df_delta,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

display( X_train )

# COMMAND ----------

# MAGIC %md ## Part 1. 分類の学習モデルを構築する

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MLflowのトラッキング
# MAGIC [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html)は、機械学習の学習コード、パラメータ、モデルを整理するためのものです。
# MAGIC 
# MAGIC また、[*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)を使用することで、MLflowの自動追跡を有効にすることができます。

# COMMAND ----------

# このノートブックのMLflowの自動ロギングを有効にする
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、MLflow の実行のコンテキスト内で分類器を訓練します。訓練されたモデル、および関連する多くのメトリクスとパラメータが自動的にログに記録されます。
# MAGIC 
# MAGIC このログには、テストデータセットにおけるモデルの AUC スコアなど、追加のメトリクスを追加することができます。

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # Models, parameters, and training metrics are tracked automatically
  model.fit(X_train, y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # The AUC score on test data is not automatically logged, so log it manually
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC このモデルの性能に満足できない場合は、ハイパーパラメータを変えて別のモデルをトレーニングしてください。

# COMMAND ----------

# Start a new run and assign a run_name for future reference
with mlflow.start_run(run_name='gradient_boost') as run:
  model_2 = sklearn.ensemble.GradientBoostingClassifier(
    random_state=0, 
    
    # Try a new parameter setting for n_estimators
    n_estimators=200,
  )
  model_2.fit(X_train, y_train)

  predicted_probs = model_2.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md ### View MLflow runs
# MAGIC ### MLflowの実行を見る
# MAGIC ログに記録されたトレーニング実行を見るには、ノートブックの右上にある **Experiment** アイコン(上部"Run all"ボタン横にある試験管のようなアイコン)をクリックし、実験サイドバーを表示します。必要に応じて、更新アイコンをクリックし、最新の実行を取得し、監視します。
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/> (画像はイメージです)
# MAGIC 
# MAGIC 実験ページアイコンをクリックすると、より詳細な MLflow 実験ページ（[AWS](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#notebook-experiments)）を表示することができ ます。このページでは、ランの比較や特定のランの詳細を確認することができます。
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのロード
# MAGIC MLflow API を使用して、特定の実行結果にアクセスすることもできます。次のセルのコードは、与えられたMLflowの実行で学習されたモデルをロードし、予測を行うためにそれを使用する方法を示しています。また、特定のモデルをロードするためのコードスニペットは、MLflow run ページで見つけることができます 
# MAGIC 
# MAGIC ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).

# COMMAND ----------

# After a model has been logged, you can load it in different notebooks or jobs
# mlflow.pyfunc.load_model makes model prediction available under a common API
model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

# モデルのロード(Sparkデータフレーム版)
from pyspark.sql.functions import struct, col
model_id = 'runs:/{run_id}/model'.format(run_id=run.info.run_id)
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_id, result_type='string')


# モデルを使って予測する際の入力データは、DeltaLakeからロードする。
# ここでは、実行テストのため、学習に使ったデータも含めてロードする。
input_data = spark.read.load('/tmp/example_codes/wine_data.delta').drop('quality')

# predict()を実行
predicted_df = input_data.withColumn('predictions', loaded_model(struct(*map(col, input_data.columns))))
display(predicted_df)

# COMMAND ----------

# Delta形式で保存(永続化)
predicted_df.write.mode('overwrite').save('/tmp/example_codes/predicted.delta')

# COMMAND ----------

# 上記で保存したファイルの確認(オプショナル)
# [方法1] Python
display(
  spark.read.load('/tmp/example_codes/predicted.delta')
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 上記で保存したファイルの確認(オプショナル)
# MAGIC -- [方法2] SQL
# MAGIC SELECT * FROM delta.`/tmp/example_codes/predicted.delta`

# COMMAND ----------

# MAGIC %md ## パート2 ハイパーパラメータのチューニング
# MAGIC この時点で、あなたは簡単なモデルを学習し、MLflowトラッキングサービスを使用して作業を整理しました。このセクションでは、Hyperoptを使用してより洗練されたチューニングを行う方法を説明します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### HyperoptとSparkTrialsによる並列学習
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/)は、ハイパーパラメータチューニングのためのPythonライブラリです。DatabricksでのHyperoptの使い方については、ドキュメント（[AWS](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/index#hyperparameter-tuning-with-hyperopt)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)）を参照してください。
# MAGIC 
# MAGIC HyperoptをSparkTrialsと併用することで、ハイパーパラメータースイープを実行し、複数のモデルを並行して学習させることができます。これにより、モデル性能の最適化に必要な時間を短縮することができます。MLflowトラッキングはHyperoptと統合されており、モデルとパラメータを自動的に記録します。

# COMMAND ----------

# Define the search space to explore
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # Enable autologging on each worker
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # Tune based on the test AUC
    # In production settings, you could use a separate validation set instead
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrials distributes the tuning using Spark workers
# Greater parallelism speeds processing, but each hyperparameter trial has less information from other trials
# On smaller clusters or Databricks Community Edition try setting parallelism=2
spark_trials = SparkTrials(
  parallelism=8
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # Use hyperopt to find the parameters yielding the highest AUC
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### ベストモデルを取得するためのランの検索
# MAGIC 全てのランはMLflowによって追跡されるので、MLflow search runs APIを使用してベストランのメトリックとパラメータを取得し、最も高いテストAUCを持つチューニングランを見つけることができます。
# MAGIC 
# MAGIC このチューニングされたモデルは、パート1で学習されたシンプルなモデルよりも良いパフォーマンスを発揮するはずです。

# COMMAND ----------

# Sort runs by their test auc; in case of ties, use the most recent run
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)
best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md ### UIで複数のランを比較する
# MAGIC Part 1 と同様に、**Experiment** サイドバーの上部にある外部リンクアイコンからアクセスできる MLflow 実験詳細ページで、ランを表示し比較することができます。
# MAGIC 
# MAGIC 実験詳細ページで、"+"アイコンをクリックして親ランを展開し、親ラン以外のすべてのランを選択して**Compare**をクリックします。平行座標プロットを使用して異なるランを視覚化し、異なるパラメータ値がメトリックに与える影響を表示することができます。
# MAGIC 
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>

# COMMAND ----------

# MAGIC %md # 999. 保存したファイルを削除(クリーンアップ)

# COMMAND ----------

# 999 クリーンアップ
dbutils.fs.rm('/tmp/example_codes/', True)

# COMMAND ----------


