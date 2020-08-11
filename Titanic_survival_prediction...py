#!/usr/bin/env python
# coding: utf-8

# #Importing the csv file

# #importing all libreries

# In[1]:


# linear algebra
import numpy as np
# data prosseing
import pandas as pd
# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier


# In[2]:


Titanic=pd.read_csv("Titanic.csv")


# In[3]:


Titanic.head()


# In[4]:


#checking the length
len(Titanic)


# In[5]:


Titanic.info()


# # survival:    Survival 
# PassengerId: Unique Id of a passenger. 
# pclass:    Ticket class     
# sex:    Sex     
# Age:    Age in years     
# sibsp:    # of siblings / spouses aboard the Titanic     
# parch:    # of parents / children aboard the Titanic     
# ticket:    Ticket number     
# fare:    Passenger fare     
# cabin:    Cabin number     
# embarked:    Port of Embarkation

# In[6]:


Titanic.describe()


# #Above we can see that 37% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.17 to 80. 

# calculating missing datas
# Titanic.isna().sum()

# In[7]:


#calculating missing datas
Titanic.isna().sum()


# In[8]:


Titanic.Sex.value_counts()


# In[9]:


#compare Target column with Sex column
pd.crosstab(Titanic.Survived,Titanic.Sex)


# In[10]:


#create a plot of survived vs sex
pd.crosstab(Titanic.Survived,Titanic.Sex).plot(kind="bar",
                                               figsize=(10,6),
                                               color=["salmon","lightblue"])
plt.title("survival frequency for sex")
plt.xlabel("0=unsurvived,1=surviver")
plt.ylabel("no of passengers")
plt.legend(["Female","Male"])


# # so,from the above graph it does make sence that the no of female survived is 75% while male survived only 14%.
# # That is we compare "Sex" column vs our target column "Survived"

# #Now it the below hist graph gives us idea about the survival vs age of passengers:-

# In[11]:


Titanic.Age.plot.hist(Titanic.Survived==1)


# # As we can see that the Cabin column has more than 75% missing data so we should drop the column
# # PassengerId and the Ticket columns should be dropped because-the first one is an arbitery integer value and the second one is for distinguished from each other
# 
# Titanic=Titanic.drop(columns=["Cabin"])

# Titanic=Titanic.drop(columns=["PassengerId","Ticket"])

# In[12]:


Titanic=Titanic.drop(columns=["Cabin"])

Titanic=Titanic.drop(columns=["PassengerId","Ticket"])
Titanic=Titanic.drop(columns=["Name"])
Titanic.head()


# In[13]:


survived='survived'
not_survived='not_survived'
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
women=Titanic[Titanic['Sex']=="female"]
men=Titanic[Titanic['Sex']=="male"]
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')


# # from both the graph we can see that for women the survival chances are higher between age 14 and 40 but for men that's between 18 to 30 
# although men have a very low chance for survival

# # values from the sex column shold be remapped to 0 and 1 instade of 'male' and 'female'

# In[14]:


Titanic.isna().sum()


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # Filling missing datas with mean and mode values
# 

# In[15]:


Titanic['Age'].fillna(Titanic['Age'].mean(),inplace=True)
Titanic['Pclass'].fillna(Titanic['Pclass'].mode()[0],inplace=True)
Titanic.head()


# In[16]:


Titanic['Pclass'].mode()


# In[17]:


Titanic.isna().sum()


# #dropping last few missing datas

# In[18]:


Titanic.dropna(axis=0,inplace=True)
len(Titanic)


# In[19]:


Titanic.isna().sum()


# In[20]:


Titanic.head()


# In[21]:


#create the corelation matrix
Titanic.corr()


# In[22]:


co_matrix=Titanic.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(co_matrix,
               annot=True,
               linewidths=0.5,
               fmt="0.2f",
               cmap="YlGnBu");


# # now the dataset is totally clean

# In[23]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_features=["Sex","Embarked","Pclass","SibSp"]
one_hot=OneHotEncoder()
transformer=ColumnTransformer([("one_hot",one_hot,categorical_features)],remainder="passthrough")


# In[24]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
np.random.seed(42)


# In[25]:


# split the whole dataset into two sets x and y
x=Titanic.drop("Survived",axis=1)
y=Titanic["Survived"]


# In[26]:


#make train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[27]:


x_train.head()


# In[28]:


y_train.head()


# In[29]:


transform_x=transformer.fit_transform(x)
transform_x


# In[30]:


pd.DataFrame(transform_x).head()


# # fit the model

# In[31]:


np.random.seed(42)
rfg=RandomForestRegressor()
# let's refil the data
x_train,x_test,y_train,y_test=train_test_split(transform_x,y,test_size=0.2,random_state=1)
rfg.fit(x_train,y_train)
rfg.score(x_test,y_test)


# # try to apply different estimators

# In[32]:


transform_x


# In[33]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
np.random.seed(1)
x=Titanic.drop("Survived",axis=1)
y=Titanic["Survived"]
x_train,x_test,y_train,y_test=train_test_split(transform_x,y,test_size=0.2)
rfc.fit(x_train,y_train)
rfc.score(x_test,y_test)


# In[ ]:





# In[34]:



from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
np.random.seed(1)
x_train,x_test,y_train,y_test=train_test_split(transform_x,y,test_size=0.2)
dt.fit(x_train,y_train)
dt.score(x_test,y_test)


# #use of cross_val_score(it split the total dataset into (cv=n) no. of section and then finds the each section score ,here we have mean of all those set and finnaly compare bothe the results)
# 
# it's good to use cross_val_score to find more accurate result
# 
# 

# In[35]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
np.random.seed(1)
x=Titanic.drop("Survived",axis=1)
y=Titanic["Survived"]
x_train,x_test,y_train,y_test=train_test_split(transform_x,y,test_size=0.2)
rfc.fit(x_train,y_train)
rfc.score(x_test,y_test)
scores=cross_val_score(rfc,transform_x,y,cv=5)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# # This looks much more realistic than before. Our model has a average accuracy of 82% with a standard deviation of 4 %. The standard deviation shows us, how precise the estimates are .
# 
# # This means in our case that the accuracy of our model can differ + — 4%.

# In[36]:


rfc= RandomForestClassifier(n_estimators=100, oob_score = True)
rfc.fit(x_train, y_train)
y_prediction = rfc.predict(x_test)

rfc.score(x_train, y_train)

acc_randomforestclassifier = round(rfc.score(x_train, y_train) * 100, 2)
print(round(acc_randomforestclassifier,2,), "%")


# In[37]:


#create a hyperparameter grid for RandomForestClassifier
rf_grid={"n_estimators":np.arange(10,1000,50),
        "max_depth":[None,3,5,10],
        "min_sample_split":np.arange(2,10,2),
        "min_sample_lead":np.arange(1,10,2)}


# In[ ]:





# In[ ]:




