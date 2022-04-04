import kfold_template
import pandas
from matplotlib import pyplot
from sklearn import svm

##SVM: Support Vector Machine
dataset = pandas.read_csv("survey_dataset_clean.csv")
dataset = dataset.sample(frac = .1).reset_index(drop = True)
target = dataset['game_categorical'].values
dataset = dataset.drop(['game_categorical'], axis = 1)
dataset = dataset.iloc[:, 1:]
data = dataset.values
print(target)
print(data)

r2_scores, accuracy_scores, confusion_matrices, confusion_2x2_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel = "linear"), 0, 1, 1)

print(r2_scores)
print(accuracy_scores)
for confusion_matrix in confusion_matrices:
	print(confusion_matrix)
for confusion_2x2_matrix in confusion_2x2_matrices:
	print(confusion_2x2_matrix)

##Predicting the games from 'new_customers_dataset.csv' using the machine built
