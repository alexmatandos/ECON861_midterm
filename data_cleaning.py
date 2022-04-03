import pandas
import os

dataset = pandas.read_csv("survey_dataset.csv")

##creating categorical variables for the variables containing multiple discrete values and a dummy variable for gender
codes, uniques = pandas.factorize(dataset['game'])
codes1, uniques1 = pandas.factorize(dataset['horoscope'])
codes2, uniques2 = pandas.factorize(dataset['region'])
codes3, uniques3 = pandas.factorize(dataset['gender'])

#print(codes)
#print(uniques)

#print(codes1)
#print(uniques1)

dataset['game_categorical'] = codes + 1
dataset['horoscope_categorical'] = codes1 + 1
dataset['region_categorical'] = codes2 + 1
dataset['gender_dummy'] = codes3

##removing the outlier values for age

#print(set(dataset['age']))
before = len(dataset)
dataset = dataset.drop(dataset[dataset['age'] < 0].index)
#print(set(dataset['age']))
after = len(dataset)

#thirty-six observations are dropped
print(before - after)

##we are also dropping the variable 'personality8' since its effect is, on average, null for the observations
dataset = dataset.drop(['personality8'], axis = 1)
print(set(codes))
print(set(uniques))
dataset.to_csv("survey_dataset_clean.csv") 

