# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
Developed by :SYED MUHAMMED ZAHI     
Regsiter no : 212221230114
~~~
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("titanic_dataset.csv")
df.info()

df.isnull().sum()

df['Cabin']=df['Cabin'].fillna(df['Cabin'].mode()[0])
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

df.skew()
df1=df.copy()
df1=df.info()
df1.skew()
df1["Sibsp_1"]=np.sqrt(df1.SibSp)
df1.SibSp.hist()
df1.skew()
df

del df['Name']
del df['Cabin']
del df['Ticket']
df.isnull().sum()

from sklearn.preprocessing import
OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder (categories =[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])
df

from category_encoders import BinaryEncoder
be1=BinaryEncoder()
df['Sex']=be1.fit_transform(df[["Sex"]])
df

#Function Transformation:

#Log Tranformation:
np.log(df["Age"])

#Reciprocal Transformation
np.reciprocal (df[["Fare"]])

#sqrt transformation
np.sqrt(df["Embarked"])

#power transformation
df["Age_boxcox"],parameters=stats.boxcox(df["Age"])
df
df["Pclass_boxcox"],parameters=stats.boxcox(df["Pclass"])
df
df["Fare_yeojohnson"],parameters = stats.yeojohnson(df["Fare"])
df
df["Parch_yeojohnson"],parameters = stats.yeojohnson(df["Parch"])
df
df.skew()

#Quantile transformation

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution ='normal',n_quantiles=891)

df["Age_1"]=qt.fit_transform(df[["Age"]])
sm.qqplot(df['Age'],line='45')

sm.qqplot(df['Age_1'],line='45')

df["Fare_1"]=qt.fit_transform(df[["Fare"]])
sm.qqplot(df["Fare"],line='45')
sm.qqplot(df['Fare_1'],line='45')

df["Parch_1"]=qt.fit_transform(df[["Parch"]])
sm.qqplot(df['Parch'],line='45')
sm.qqplot(df['Parch_1'],line='45')
df
~~~

# Data_for_Transform :
~~~
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

#Function Transformation 
#Log Transformation 
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])

df["Highly positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df["Moderate Positive Skew_yeojohnson"],parameters=stats.boxcox(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()
#Quantile Transformation 
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution ='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df["Highly Positive Skew"],line='45')

df
~~~

# OUPUT
![image](https://user-images.githubusercontent.com/94187572/168719706-4ee816c5-8fda-4e3a-a42a-d5825da0d996.png)
![image](https://user-images.githubusercontent.com/94187572/168719797-9dc14102-769c-4585-a7ed-297d28a7edd4.png)
![image](https://user-images.githubusercontent.com/94187572/168719826-c42bdc2d-ab01-425f-b33d-0b97e7c0d1af.png)
![image](https://user-images.githubusercontent.com/94187572/168719858-acdb5178-0a8b-4e47-8801-5da4c0eca56c.png)
![image](https://user-images.githubusercontent.com/94187572/168719895-289d37d7-aa1d-4447-826a-970165e5fa2e.png)
![image](https://user-images.githubusercontent.com/94187572/168719908-5951cc43-47ee-4583-96c0-624493df751d.png)
![image](https://user-images.githubusercontent.com/94187572/168719932-f67bed06-1dc4-4e7d-83d9-732b50ece9ce.png)

# RESULT:
The various feature transformation techniques on a dataset and save the data to a file has been performed successfully.




