import numpy as np                                                 # Lib to deal with array
import pandas as pd                                                # Lib to deal with data
import matplotlib.pyplot as plt                                    # Lib to plot graphs
from sklearn import linear_model                                   # Lib contain linear regression model

df = pd.read_csv("J:\machine_learning\plot.csv")                   # using panda read a csv file 
df                                                                 # print the csv file



%matplotlib inline                                                 # Using this magic to plot graph
plt.xlabel('Area')                                                 # Providing the label in X axis
plt.ylabel('Price')                                                # Providing the label in Y axis
plt.scatter(df.Area,df.Price,color='red',marker='+')               # Printing the graph


reg = linear_model.LinearRegression()                              # Making a object of linearRegression from linear_model
reg.fit(df[['Area']],df.Price)                                     # Now finding the line fitted the most by finding sigma(delta(e)^2)

reg.predict(150)                                                   # find the predicted price for this area

                            # [Y = m X + C], the above fit find the best 'm' and best 'c' and we already have x = 150 for this case

reg.coef_                                                          # give the value of 'm'
reg.intercept_                                                     # give the value of 'c'