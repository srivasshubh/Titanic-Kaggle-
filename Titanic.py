
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


x = pd.read_csv('train.csv')
x.describe()


# In[3]:


x = pd.read_csv('C:\Users\Shubham Srivastava\Desktop\Python_ML\train.csv')
x.describe()


# In[4]:


import os
os.chdir("C:\Users\Shubham Srivastava\Desktop\Python_ML")
x = pd.read_csv("train.csv")
x.describe()


# In[5]:


import os
os.chdir("C:\Users\Shubham Srivastava\Desktop\Python_ML")
x = pd.read_csv("train.csv")
x.describe()


# In[6]:


import os
import pandas as pd
os.chdir("C:\Users\Shubham Srivastava\Desktop\Python_ML")
data = pd.read_csv("workbook1.csv")


# In[7]:


import os
os.path.isfile('C:\Users\Shubham Srivastava\Desktop\Python_ML\train.csv')


# In[8]:


import os
os.path.isfile('/Users/Shubham Srivastava/Desktop/Python_ML/train.csv')


# In[9]:


import os
os.path.isfile('/Users/Shubham Srivastava/Desktop/Python_ML/train.csv')


# In[10]:


import os
os.path.isfile('C:/Users/Shubham Srivastava/Desktop/Python_ML/train.csv')


# In[11]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[12]:


x = pd.read_csv('C:/Users/Shubham Srivastava/Desktop/Python_ML/train.csv')
x.describe()


# In[13]:


y = x.pop('Survived')
y.head()


# In[14]:


y.describe()


# In[15]:


numeric_variabe = list(x.dtype[x.dtype != 'object'].index)
x[numeric_variable].head()


# In[16]:


numeric_variabe = list(x.dtypes[x.dtypes!= 'object'].index)
x[numeric_variable].head()


# In[17]:


numeric_variable = list(x.dtype[x.dtype != 'object'].index)
x[numeric_variable].head()


# In[18]:


numeric_variable = list(x.dtypes[x.dtypes != 'object'].index)
x[numeric_variable].head()


# In[19]:


model = RandomForestClassifier(n_estimator=100)
model.fit(x[numeric_variable],y)


# In[20]:


model = RandomForestClassifier(n_estimators=100)
model.fit(x[numeric_variable],y)


# In[21]:


model = RandomForestClassifier()
model.fit(x[numeric_variable],y)


# In[22]:


x['Age'].fillna(x.age.mean(), implace=true)


# In[23]:


x['Age'].fillna(x.Age.mean(), implace=true)


# In[24]:


x['Age'].fillna(x.age.mean(), implace=true)


# In[25]:


x['Age'].fillna(x.Age.mean(), implace=true)


# In[26]:


x['Age'].fillna(x.Age.mean(), implace=True)


# In[27]:


x['Age'].fillna(x.Age.mean(), inplace=True)


# In[28]:


model = RandomForestClassifier(n_estimators=100)
model.fit(x[numeric_variable],y)


# In[29]:


model = RandomForestClassifier(n_estimators=100)
model.fit(x[numeric_variable],y)


# In[31]:


print("Train Accuracy :: ", accuracy_score(y, model.predict(x[numeric_variable])))


# In[32]:


test = pd.read_csv('C:/Users/Shubham Srivastava/Desktop/Python_ML/test.csv')


# In[33]:


test[numeric_variable].head()


# In[34]:


test['Age'].fillna(test.Age.mean(), inplace = True)


# In[35]:


y_pred = model.predict(test[numeric_variable])


# In[36]:


test= test[numeric_variable].fillna(test.mean().copy())


# In[37]:


y_pred = model.predict(test[numeric_variable])


# In[38]:


print(y_pred)


# In[40]:


submission = pd.DataFrame({"PassengerId" : test["PassengerId"], "Survived" : y_pred})


# In[41]:


submission.to_csv('C:/Users/Shubham Srivastava/Desktop/Python_ML/Titanic/titanic_ans.csv', index = False)


# In[42]:


submission.to_csv('C:/Users/Shubham Srivastava/Downloads/titanic_ans.csv', index = False)

