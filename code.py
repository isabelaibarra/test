echo "# test" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/isabelaibarra/test.git
git push -u origin main


#loading dataset
import pandas as pd
import numpy as np
import shap #for SHAP values


#visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# data splitting
from sklearn.model_selection import train_test_split

# data modeling
from sklearn.tree import DecisionTreeClassifier


from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import export_graphviz #plot tree





dt = pd.read_csv("SupervisedLearning/HeartDiseaseIdentification/heart.csv")

dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
dt = pd.get_dummies(dt, drop_first = True)




from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .2, random_state=10) #split the data

model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'Deceased'
y_train_str[y_train_str == '1'] = 'Survived'
y_train_str = y_train_str.values

export_graphviz(estimator, out_file='SupervisedLearning/HeartDiseaseIdentification/heart.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'heart.dot', '-o', 'heart.png', '-Gdpi=600'])



from IPython.display import Image
Image(filename = 'SupervisedLearning/HeartDiseaseIdentification/heart.png')
