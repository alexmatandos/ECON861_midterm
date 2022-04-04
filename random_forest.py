import pandas
import kfold_template
from sklearn.ensemble import RandomForestClassifier

##Random Forest
dataset = pandas.read_csv("survey_dataset_clean.csv")
dataset = pandas.get_dummies(dataset)

dataset=dataset.sample(frac = 1).reset_index(drop = True)

print(dataset)

target = dataset['game_categorical'].values
data_raw = dataset.drop('game_categorical', axis = 1)
data_raw = data_raw.iloc[:, 1:]
data = data_raw.values
print(target)
print(data)

machine = RandomForestClassifier(criterion = "gini", max_depth = 10, n_estimators = 300 ,bootstrap = True, max_features = "auto")
r2_scores, accuracy_scores, confusion_matrices, confusion_2x2_matrices = kfold_template.run_kfold(data, target, 4, machine, 0, 1, 1)
print(r2_scores)
print(accuracy_scores)
for confusion_matrix in confusion_matrices:
	print(confusion_matrix)
for confusion_2x2_matrix in confusion_2x2_matrices:
	print(confusion_2x2_matrix)

##Predicting the games from 'new_customers_dataset.csv' using the machine built
validation = pandas.read_csv("new_customers_dataset_clean.csv")
validation = validation.iloc[:, 1:]
prediction = machine.predict(validation.values)
print(prediction)

validation['prediction'] = prediction
validation.to_csv("new_customers_dataset_clean_forecast.csv")