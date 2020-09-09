#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from pydataset import data


# In[5]:


from env import host, user, password


# In[6]:


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In a jupyter notebook, classification_exercises.ipynb, use a python module (pydata or seaborn datasets) containing datasets as a source from the iris data. Create a pandas dataframe, df_iris, from this data.
# <ul>
# <li>print the first 3 rows
# <li>print the number of rows and columns (shape)
# <li>print the column names
# <li>print the data type of each column
# <li>print the summary statistics for each of the numeric variables. Would you recommend rescaling the data based on these statistics?</ul>

# In[7]:


iris = pd.read_sql('SELECT * FROM measurements', get_connection('iris_db'))


# In[8]:


iris.to_csv('iris.csv')


# In[25]:


def get_iris_data():
    filename = "iris.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM measuresments', get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)

        # Return the dataframe to the calling code
        return df  


# In[13]:


df_iris = pd.read_csv('iris.csv', index_col=0)
df_iris.head(3)


# In[15]:


df_iris.info()


# In[16]:


for col in df_iris.columns: 
    print(col) 


# In[19]:


df_iris.dtypes
#based off the data types i would convert the measurement and species id from an integer to a float


# Read the Table1_CustDetails table from the Excel_Exercises.xlsx file into a dataframe named df_excel.
# <ol>
# <li> assign the first 100 rows to a new dataframe, df_excel_sample
# <li>print the number of rows of your original dataframe
# <li>print the first 5 column names
# <li>print the column names that have a data type of object
# <li>compute the range for each of the numeric variables.
#     </ol>

# In[29]:


excel_df = pd.read_csv('telco.csv', index_col=0)
excel_df.head(100)


# In[22]:


titanic=pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))


# In[23]:


titanic.to_csv('titanic.csv')


# In[26]:


def get_titanic_data():
    filename = "titanic.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)

        # Return the dataframe to the calling code
        return df  


# In[ ]:




