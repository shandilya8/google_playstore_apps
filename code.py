# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:00:55 2021

@author: shandilya
"""


#Google play store apps

#import starndar libraries
import numpy as np
import pandas as pd
#import Useful dependencies
import seaborn as sns
import matplotlib.pyplot as plt

#data
data=pd.read_csv("D:/Project-Apps/DataSet/googleplaystore.csv")

# print(Data.head(10))
# print(Data.dtypes)
print("Data Shape",data.shape)

print(data.info())

print("NUll values in each attribute:","\n",data.isnull().sum())

#------------------------------------------
#Data Cleaning

reviews=data[data['Reviews']=='3.0M']
# print(reviews)


#data.loc[data[data['Category']=='1.9'].index[0]]
#output of this will be 
#App               Life Made WI-Fi Touchscreen Photo Frame
# Category                                              1.9
# Rating                                                 19
# Reviews                                              3.0M
# Size                                               1,000+
# Installs                                             Free
# Type                                                    0
# Price                                            Everyone
# Content Rating                                        NaN
# Genres                                  February 11, 2018
# Last Updated                                       1.0.19
# Current Ver                                    4.0 and up
# Android Ver                                           NaN
# Name: 10472, dtype: object


for i in range(len(data.columns)-1,1,-1):
    data.loc[[data[data['Category']=='1.9'].index[0]],[data.columns[i]]]=data.loc[[data[data['Category']=='1.9'].index[0]],[data.columns[i-1]]].values
data.loc[[data[data['Category']=='1.9'].index[0]],['Category']]='ART_AND_DESIGN'

# print(data.iloc[10472])

#converting Review type to int
data["Reviews"]=data['Reviews'].astype(int)


data['Price'].replace(to_replace='0',value='$0',inplace=True)
data['Price']=data['Price'].apply(lambda a : a[1:])
data['Price']=data['Price'].astype(float)

data['Size'].replace(to_replace='Varies with device',value='0M',inplace=True)
data['Size']=data['Size'].apply(lambda a : a.replace(',',''))

data['Value']=data['Size'].apply(lambda a : a[:-1])# taking on values from size attribute
data['Unit']=data['Size'].apply(lambda a : a[-1:])#taking only M from the size attribute
data['Value']=data['Value'].astype(float)

data['Installs'].replace(to_replace='0',value='0+',inplace=True)
data['Installs']=data['Installs'].apply(lambda a: a.replace(',',''))
data['Installs']=data['Installs'].apply(lambda a : a[:-1])
data['Installs']=data['Installs'].astype(int)

data['Rating']=data['Rating'].astype('float')

#boxplot
plt.figure()
sns.boxplot(x='Category',y='Rating',data=data)
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.boxplot(x='Content Rating',y='Rating',data=data)
plt.show()

print("\n")
print("Frequency of Content rating :")
print(data['Content Rating'].value_counts())
print('\n')

# print(data[data['Genres'].isnull()])
#filling null value with 
data.loc[[data[data['Genres'].isnull()].index[0]],['Genres']]='Art & Design'

# print(data[data['Type'].isnull()])
#filling null values with mode of type attribute
data['Type'].fillna(data['Type'].mode()[0],inplace=True)
# print(data['Type'].isnull().sum())

# print(data[data['Android Ver'].isnull()])
data['Android Ver'].fillna(data['Android Ver'].mode()[0],inplace=True)
# print(data['Android Ver'].isnull().sum())

# print(data[data['Current Ver'].isnull()])

temp=pd.DataFrame()
for i in data['Category'].unique():
    temp1=data[(data['Category']==i)]['Rating'].fillna(data[(data['Category']==i)]['Rating'].mode()[0])
    temp=pd.concat([temp,temp1])
data['Rating']=temp

# print(data.info())

plt.figure()
sns.scatterplot(x=data['Reviews'],y=data['Rating'])
plt.show()

# plt.scatter(data['Reviews'], data['Rating'])
# plt.show()

plt.Figure()
sns.scatterplot(x='Price',y='Reviews',data=data)
plt.show()

plt.figure()
sns.scatterplot(x='Price',y='Rating',data=data)
plt.show()

plt.figure()
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.show()

#-----------App with largest size ----------------------

#app size 
app_size=data[['App','Size','Value','Unit']]

sns.countplot(app_size['Unit'])
plt.show()

# print(app_size.head())

print("Apps with largest size are :")

#sorting values of App and Size
Largest=app_size[app_size['Unit']=='M'].sort_values('Value',ascending=False)[['App','Size']].head()
print(Largest)
print('\n')


#-----------App with Largest Number of Installation-----------------

app_inst=data[['App','Installs']]
#print(app_inst.head())

print("Apps with largest Installations are:")

L_inst=app_inst.sort_values('Installs',ascending=False).head()
print(L_inst)
print("\n")

#------------------Most Popular Category ---------------

app_grp = data.groupby('Category')

grp=app_grp.mean()
grp['Count']=app_grp['App'].count()#counting the number of categories in Category attribute 
# print(grp["Count"])



#-----------------Popularity based on customer Preference----------------

print("Popularity based on Customer preference :")
popu=grp.sort_values(['Reviews','Rating'],ascending=False).head()
print(popu)
print('\n')


#-----------------Popularity base on App development--------------

print("Popularity based on app development: ")
pop_app_dev=grp.sort_values(['Count'],ascending=False).head()
print(pop_app_dev)
print('\n')

#--------------------Paid vs Free------------
plt.figure()
sns.countplot(x='Type',data=data)
plt.show()

plt.figure()
sns.histplot(x='Rating',hue='Type',data=data,kde=True)
plt.show()
print("\n")

type_grp=data.groupby('Type')
type_grp=type_grp.mean()
type_grp['Count']=data.groupby('Type')['App'].count()

print(type_grp)
print('\n')
# print("Paid VS Free")
plt.figure()
ax=sns.barplot(x=type_grp.index,y='Rating',data=type_grp)
ax.set_title("Paid Vs Free")
plt.show()


#------------Most Downloads on the latest update year 2010 to 2018-------------

app_year=data[['App','Installs','Reviews','Rating','Last Updated']]
app_year['Year']=app_year['Last Updated'].apply(lambda a : a[-4:])
app_year['Year']=app_year['Year'].astype(int)

# print(app_year.head())

year_grp=app_year.sort_values(['Installs','Reviews'],ascending=False).groupby('Year')
year_grp.first().sort_index(ascending=False)

year_grp=app_year.sort_values(['Reviews','Installs'],ascending=False).groupby('Year')
year_grp.first().sort_index(ascending=False)




#----------------Prediction Model----------------


#------------------pre-processing----------
from sklearn.preprocessing import LabelEncoder

#total 15 attributes
to_delete=data[['Size','Type','Current Ver','Android Ver']] #4 attributes
required_data=data[['App','Category','Reviews','Installs','Price','Content Rating','Value','Unit','Genres']]#9 attributes
target=data['Rating']
target.index=data['App'] #target data with index as App attribute

required_data['Year']=data['Last Updated'].apply(lambda a : a[-4:])
required_data['Year']=required_data['Year'].astype(int)

required_data.set_index('App',inplace=True)
# print(required_data.head())

#.get_dummies converts categorical data into dummy or indicator variables
required_data=pd.get_dummies(required_data,columns=['Unit'],drop_first=True)

LE=LabelEncoder()
LE.fit(required_data['Content Rating'])
required_data['Content Rating']=LE.transform(required_data['Content Rating'])

LE.fit(required_data['Category'])
required_data['Category']=LE.transform(required_data['Category'])

LE.fit(required_data['Genres'])
required_data['Genres']=LE.transform(required_data['Genres'])

# print(required_data.head(5))
# print(required_data.info())
print(required_data.shape)


#---------------Model Creation and Prediction------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

sc=StandardScaler()
sc.fit(required_data)
required_data_sc=sc.transform(required_data)

X_train, X_test, Y_train, Y_test=train_test_split(required_data_sc,target,test_size=0.3)

prediction= pd.DataFrame(Y_test)
# print(prediction.head())

LR=LinearRegression()
LR.fit(X_train,Y_train)

prediction['Linear_R']=LR.predict(X_test)
prediction['Linear_R']=round(prediction['Linear_R'],1)

#RMSE between predicted Y_test and predicted X_test
print(np.sqrt(mean_squared_error(prediction['Rating'],prediction['Linear_R'])))

#lasso()
lasso=Lasso()
lasso.fit(X_train,Y_train)

prediction['Lasso']=lasso.predict(X_test)
prediction['Lasso']=round(prediction['Lasso'],1)

print(np.sqrt(mean_squared_error(prediction['Rating'], prediction['Lasso'])))

#print(prediction.head())


#---------------Recommendation Sysytem---------

app_recom=data[(data['Installs']>1000) & (data['Reviews']>100)][['Category','Reviews','Rating','Installs','Price','Content Rating','Genres']]

app_recom=pd.get_dummies(app_recom,columns=['Category','Content Rating','Genres'],prefix='',prefix_sep='')

sc=StandardScaler()
sc.fit(app_recom)
app_recom=pd.DataFrame(sc.transform(app_recom))

app_recom.index=data[(data['Installs']>1000) & (data['Reviews']>100)]['App']
app_recom=app_recom.T
# print(app_recom.head())
# print(app_recom.shape)

for_app=app_recom['Do Not Crash']

recommend=pd.DataFrame(app_recom.corrwith(for_app),columns=['Correlation'])
print(recommend.sort_values('Correlation',ascending=False).head())







