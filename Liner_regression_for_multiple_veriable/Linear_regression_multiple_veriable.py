import numpy as np
import pandas as pd
import math
from sklearn import linear_model


df = pd.read_csv("J:\machine_learning\plot_multiple_veriable.csv")



bedroom_median = math.floor(df.Bedroom.median())                   # finding the median for bedroom


df.Bedroom = df.Bedroom.fillna(bedroom_median)                     # filling the meadian to the NaN


reg = linear_model.LinearRegression()                              # creating regression object
reg.fit(df[['Area','Bedroom','Age']],df.Price)                     # train the data


reg.predict([[3000,4,40]])                                         # prdict the value


reg.coef_                                                          # Give 3 value 'm1', 'm2', 'm3'
reg.intercept_                                                     # give 'c'




                       # formula =>  Y = m1 * area + m2 * Bedroom + m3 * Age + c 