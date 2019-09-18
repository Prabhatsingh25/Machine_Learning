import numpy as np
import pandas as pd
import pickle                                                                          # Use to save the train model so that we
                                                                                       # don't have to train it again and again
from sklearn import linear_model

df = pd.read_csv("J:\Machine_Learning\Show_the_use_of_pickle\plot.csv")

reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df.Price)

with open('J:\Machine_Learning\Show_the_use_of_pickle\pickle_save_model','wb') as f:   # The way to save the train model in a file
    pickle.dump(reg,f)                                                                 # name 'pickle_save_model' uing dump method



with open('J:\Machine_Learning\Show_the_use_of_pickle\pickle_save_model','rb') as f:   # The way to retrieve the train model from a file
    model = pickle.load(f)                                                             # name 'pickle_save_model' uing lode method



model.predict(2500)                                                                    # Output using the save model array([27087.93115683])

reg.predict(2500)                                                                      # Output using the actual model array([27087.93115683])