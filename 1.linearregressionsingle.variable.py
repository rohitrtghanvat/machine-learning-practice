# https://youtu.be/8jazNUpO3lQ?si=oN8nAwD4F02mkopM

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the data from an Excel file
df = pd.read_excel(r"E:\ml\codeof ml\Book2.xlsx")  # Use read_excel() for Excel files

# Visualizing the data
plt.xlabel('area')
plt.ylabel('price(US$)')
plt.scatter(df['area'], df['price'], color='red', marker='+')


# Fitting the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Predict the price for a given area (3300)
prediction = reg.predict([[3300]])  # Reshape the scalar to 2D array
print("Predicted price for area 3300:", prediction[0])
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()


# #book1
# b = pd.read_excel(r"C:\Users\ROHIT\OneDrive\Desktop\Book1.xlsx")


# p=reg.predict(b)
# b['prices']=p
# print (b)

# b.to_excel(r"C:\Users\ROHIT\OneDrive\Desktop\Book1.xlsx")
