import kfold_template
import pandas
from matplotlib import pyplot
from sklearn import svm

#SVM: Support Vector Machine
dataset = pandas.read_csv("survey_dataset_clean.csv")

target = dataset.iloc[:, 15].values
data = dataset.iloc[:, 5:12].values

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel = "linear"), 1, 1)

print(r2_scores)
print(accuracy_scores)
print(confusion_matrices)