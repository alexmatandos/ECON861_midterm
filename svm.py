import kfold_template
import pandas
from matplotlib import pyplot
from sklearn import svm

##SVM: Support Vector Machine
dataset = pandas.read_csv("survey_dataset_clean.csv")
target = dataset['game_categorical'].values
dataset = dataset.drop(['game_categorical'], axis = 1)
data = dataset.values

r2_scores, accuracy_scores, confusion_matrices, confusion_2x2_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel = "linear"), 1, 1, 1)

print(r2_scores)
print(accuracy_scores)
for confusion_matrix in confusion_matrices:
	print(confusion_matrix)
for confusion_2x2_matrix in confusion_2x2_matrices:
	print(confusion_2x2_matrix)

##Predicting the games from 'new_customers_dataset.csv' using the machine built
validation = pandas.read_csv("new_customers_dataset.csv")
validation = dataset.drop(validation[validation['age'] < 0].index)
validation = validation.drop(['personality8'], axis = 1)
validation = pandas.get_dummies(validation)

prediction = machine.predict(validation)
print(validation)
