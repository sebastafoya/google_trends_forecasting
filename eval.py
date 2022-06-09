import pandas as pd
import ast
import json

df = pd.read_csv("data/models_results_groups_12.csv")

# Cast string column to list
df.feature_set = df.feature_set.apply(lambda x: ast.literal_eval(x))

# Check if google trends are used
df['google_trends'] = df.apply(lambda x: x.feature_set[0] != 'lag_28', axis=1)

df.errors = df.errors.apply(lambda x: json.loads(x))

df['smape'] = df.errors.apply(lambda x: x['SMAPE'])
df['mae'] = df.errors.apply(lambda x: x['MAE'])
df['md'] = df.errors.apply(lambda x: x['MD'])

trends_df = df[df.google_trends == True]
no_trends_df = df[df.google_trends == False]

idx = trends_df.groupby(['object_id', 'model', ])['rmse'].transform(max) == trends_df['rmse']

results_trends = trends_df.groupby(['object_id', 'model'])['rmse', 'smape', 'mae', 'md'].min().reset_index()
results_no_trends = no_trends_df.groupby(['object_id', 'model'])['rmse', 'smape', 'mae', 'md'].min().reset_index()

results_trends = results_trends.groupby(['model'])['rmse', 'smape', 'mae', 'md'].mean().reset_index()
results_trends.model = results_trends.model + ' - Google Trends'
results_no_trends = results_no_trends.groupby(['model'])['rmse', 'smape', 'mae', 'md'].mean().reset_index()

results_trends.to_csv('data/results_trends.csv', index=False)
results_no_trends.to_csv('data/results_no_trends.csv', index=False)


