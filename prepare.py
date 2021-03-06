#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

import acquire


# The end product of this exercise should be the specified functions in a python script named prepare.py. Do these in your classification_exercises.ipynb first, then transfer to the prepare.py file.
# 
# This work should all be saved in your local classification-exercises repo. Then add, commit, and push your changes.
# 
# Using the Iris Data:
# 
# Use the function defined in acquire.py to load the iris data.
# 
# Drop the species_id and measurement_id columns.
# 
# Rename the species_name column to just species.
# 
# Create dummy variables of the species name.
# 
# Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

# In[3]:


# Use the function defined in acquire.py to load the iris data.
iris = acquire.get_iris_data()
iris.head()


# In[4]:


# Drop the species_id and measurement_id columns.
cols_to_drop = ['species_id', 'measurement_id']
iris = iris.drop(columns=cols_to_drop)


# In[5]:


iris.columns


# In[6]:


# Rename the species_name column to just species.
iris= iris.rename(columns={'species_name': 'species'})
iris=iris.rename(columns={'species_id.1': 'speciesid'})
iris


# In[7]:


# Create dummy variables of the species name.
iris.species.value_counts()
iris_dummies = pd.get_dummies(iris[['species']], drop_first=[True, True])
iris = pd.concat([iris, iris_dummies], axis=1)
iris


# In[8]:


def iris_prep(cached=True):
    df = acquire.get_iris_data()
    df = df.drop(columns=['species_id', 'species_id.1']).rename(columns={'species_name': 'species'})
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    df = pd.concat([df, species_dummies], axis=1)
    return df


# In[9]:


prepped = iris_prep()
prepped.sample(3)


# In[26]:


titanic = acquire.get_titanic_data()
titanic.head()


# In[27]:


##handling nulls
titanic[titanic.embark_town.isnull()]
titanic[titanic.embarked.isnull()]


# In[28]:


titanic = titanic[~titanic.embarked.isnull()]
titanic.info()


# In[29]:


## removing the deck column
titanic = titanic.drop(columns='deck')


# In[30]:


## Create a dummy variable of the embarked column.
titanic_dummies = pd.get_dummies(titanic.embarked, drop_first=True)
titanic_dummies.sample(5)


# In[31]:


titanic = pd.concat([titanic, titanic_dummies], axis=1)
titanic.head()


# In[16]:


## split data
train_validate, test = train_test_split(titanic, test_size=.2, 
                                        random_state=123, 
                                        stratify=titanic.survived)


# In[17]:


train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)


# In[18]:


print(f'train:{train.shape}')
print(f'validate: {validate.shape}')
print(f'test: {test.shape}')


# In[19]:


## create a function to do the same thing
def titanic_split(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test

train, validate, test = titanic_split(titanic)


# In[20]:


print(f'train:{train.shape}')
print(f'validate: {validate.shape}')
print(f'test: {test.shape}')


# In[21]:


## helper function to impute age
def impute_mean_age(train, validate, test):
    imputer = SimpleImputer(strategy = 'mean')
    train['age'] = imputer.fit_transform(train[['age']])
    validate['age'] = imputer.transform(validate[['age']])
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


# In[22]:


def titanic_prep(cached=True):
    df = acquire.get_titanic_data()
    df = df[~df.embarked.isnull()]
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    df = pd.concat([df, titanic_dummies], axis=1)
    df = df.drop(columns='deck')
    train, validate, test = titanic_split(df)
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test


# In[23]:


train, validate, test = titanic_prep()


# In[24]:


print(f'train:{train.shape}')
print(f'validate: {validate.shape}')
print(f'test: {test.shape}')


# In[1]:


def clean_data():
    df.drop_duplicates(inplace=True)
    df.drop(columns=['deck', 'embarked', 'class', 'age'], inplace=True)
    df.embark_town.fillna(value='Southampton', inplace=True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    return pd.concat([df, dummy_df], axis=1)


# In[2]:


def prep_titanic_data():
    df = clean_data()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    train, validate, test = impute_mode()
    return train, validate, test

