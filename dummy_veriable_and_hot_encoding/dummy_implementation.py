import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('J:\Machine_Learning\dummy_veriable_and_hot_encoding\state_land_cost.csv')

dummies = pd.get_dummies(df.City)                                                              # it will create a dummy you can see in the picture provided in the file with name 'dummies'

new_df = pd.concat([df,dummies],axis='columns')                                                # it will concate the dummy with with data frame

df_after_rows_drop = new_df.drop(['City','jharkhand'],axis='columns')                          # IMPORTANT: while removing the column for whom we are creating dummy we should remove any one of the dummy column
                                                                                               # you can see it in picture with name 'df_after_rows_drop'


x = df_after_rows_drop.drop(['Price'],axis='columns')                                          # create test data

y = df_after_rows_drop.Price                                                                   # create train data


reg = linear_model.LinearRegression()
reg.fit(x,y)

reg.predict([[1100,1,0]])


reg.score(x,y)                                                                              #  the total accuracy 0.7871035247658086