import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('titanic_train.csv')
X_train = dataset
X_train = X_train.drop('Survived',axis=1)
X_train = X_train.drop('Name',axis=1)
X_train = X_train.drop('Sex',axis=1)
X_train = X_train.drop('Ticket',axis=1)
X_train = X_train.drop('Cabin',axis=1)
X_train = X_train.drop('Embarked',axis=1)
X_train['SibSp'] = X_train['SibSp'] + X_train['Parch']
X_train = X_train.drop('Parch',axis=1)

X_train = X_train.iloc[:,:].values

#Missing Values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer = imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.fit_transform(X_train[:,2:3])

#Normalization
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
X_train[:,4:5] = standard_scaler_x.fit_transform(X_train[:,4:5])

df_sex = pd.get_dummies(dataset['Sex'])
embar = dataset['Embarked']
embar = embar.fillna('S')
df_embar = pd.get_dummies(embar)

cab = dataset['Cabin']
cab = cab.str[0]
cab = cab.fillna('Z')
cab = cab.replace('T','Z')
df_cab = pd.get_dummies(cab)

naam = dataset['Name']
naam = dataset.Name.apply(lambda x: x.split(', ')[1])
naam = naam.to_frame()
naam = naam.Name.apply(lambda x: x.split('. ')[0])
naam = naam.replace('Capt','Mr')
naam = naam.replace('Col','Mr')
naam = naam.replace('Don','Mr')
naam = naam.replace('Dr','Mr')
naam = naam.replace('Jonkheer','Mr')
naam = naam.replace('Lady','Mrs')
naam = naam.replace('Major','Mr')
naam = naam.replace('Mlle','Mrs')
naam = naam.replace('Mme','Miss')
naam = naam.replace('Ms','Mr')
naam = naam.replace('Rev','Mr')
naam = naam.replace('Sir','Mr')
naam = naam.replace('the Countess','Mr')
df_name = pd.get_dummies(naam)

X_train = np.c_[X_train,df_cab.values]
X_train = np.c_[X_train,df_embar.values]
X_train = np.c_[X_train,df_sex.values]
X_train = np.c_[X_train,df_name.values]

y = dataset['Survived'].values


dataset_test = pd.read_csv('titanic_test.csv')
X_test = dataset_test
X_test = X_test.drop('Name',axis=1)
X_test = X_test.drop('Sex',axis=1)
X_test = X_test.drop('Ticket',axis=1)
X_test = X_test.drop('Cabin',axis=1)
X_test = X_test.drop('Embarked',axis=1)
X_test['SibSp'] = X_test['SibSp'] + X_test['Parch']
X_test = X_test.drop('Parch',axis=1)

X_test = X_test.iloc[:,:].values

#Missing Values
from sklearn.preprocessing import Imputer
imputer2 = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer2 = imputer2.fit(X_test[:,2:3])
X_test[:,2:3] = imputer2.fit_transform(X_test[:,2:3])

imputer3 = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer3 = imputer3.fit(X_test[:,4:5])
X_test[:,4:5] = imputer3.fit_transform(X_test[:,4:5])

#Normalization
from sklearn.preprocessing import StandardScaler
standard_scaler_t = StandardScaler()
X_test[:,4:5] = standard_scaler_t.fit_transform(X_test[:,4:5])


df_sex_test = pd.get_dummies(dataset_test['Sex'])
embar_test = dataset_test['Embarked']
embar_test = embar_test.fillna('S')
df_embar_test = pd.get_dummies(embar_test)

naam_test = dataset_test['Name']
naam_test = dataset_test.Name.apply(lambda x: x.split(', ')[1])
naam_test = naam_test.to_frame()
naam_test = naam_test.Name.apply(lambda x: x.split('. ')[0])

naam_test = naam_test.replace('Col','Mr')
naam_test = naam_test.replace('Dona','Mr')
naam_test = naam_test.replace('Dr','Mr')
naam_test = naam_test.replace('Ms','Mr')
naam_test = naam_test.replace('Rev','Mr')

df_name_test = pd.get_dummies(naam_test)

cab_test = dataset_test['Cabin']
cab_test = cab_test.str[0]
cab_test = cab_test.fillna('Z')
df_cab_test = pd.get_dummies(cab_test)


X_test = np.c_[X_test,df_cab_test.values]
X_test = np.c_[X_test,df_embar_test.values]
X_test = np.c_[X_test,df_sex_test.values]
X_test = np.c_[X_test,df_name_test.values]

#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(100,oob_score=True,n_jobs=-1,random_state=42)
#model.fit(X_train,y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y)

y_pred = model.predict(X_test)


ans2 = pd.DataFrame({"PassengerId" : dataset_test["PassengerId"], "Survived" : y_pred})
ans2.to_csv('ans3.csv', index=False)
