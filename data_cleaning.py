import pandas
import os

dataset = pandas.read_csv("survey_dataset.csv")

##creating categorical variables for the variables containing multiple discrete values
codes, uniques = pandas.factorize(dataset['game'])

#print(codes)
#print(uniques)

#print(codes1)
#print(uniques1)

dataset['game_categorical'] = codes + 1

##removing the outlier values for age

#print(set(dataset['age']))
before = len(dataset)
dataset = dataset.drop(dataset[dataset['age'] < 0].index)
#print(set(dataset['age']))
after = len(dataset)

#thirty-six observations are dropped
print(str(before - after) + " observations are dropped")

##we are also dropping the variable 'personality8' since its effect is, on average, null for the observations and also we create dummy variables for all the independent categorical variables values
dataset = dataset.drop(['personality8'], axis = 1)
dataset = pandas.get_dummies(dataset)

##dropping the columns for the redundant dummies for games
dataset = dataset.drop(['game_Overwatch'], axis = 1)
dataset = dataset.drop(['game_Diablo'], axis = 1)
dataset = dataset.drop(['game_Hearthstone'], axis = 1)
dataset = dataset.drop(['game_The Lost Vikings'], axis = 1)
dataset = dataset.drop(['game_Warcraft'], axis = 1)
dataset = dataset.drop(['game_Starcraft'], axis = 1)
dataset = dataset.drop(['game_Heroes of the Storm'], axis = 1)

dataset.to_csv("survey_dataset_clean.csv") 

