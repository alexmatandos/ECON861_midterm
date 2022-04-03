import pandas
import kfold_template
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("survey_dataset_clean.csv")
dataset = pandas.get_dummies(dataset)

dataset=dataset.sample(frac = 1).reset_index(drop = True)

print(dataset)

target = dataset['game_categorical'].values
data_raw = dataset.drop('game_categorical', axis = 1)
data = data_raw.values

feature_list = data_raw.columns


print(target)
print(data)

machine = RandomForestClassifier(criterion = "gini", max_depth = 10, n_estimators = 300 ,bootstrap = True, max_features = "auto")
results = kfold_template.run_kfold(data, target, 4, machine, 0, 0, 1)
print(results)
