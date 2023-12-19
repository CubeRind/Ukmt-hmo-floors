#MACHINE LEARNING - SUPERVISED LEARNING - REGRESSION - LINEAR REGRESSION

#UNIVARIATE/SIMPLE/SINGLE - 1 column as input and 1 column as output
#MULTIVARIATE/MULTIPLE - Many columns as input and 1 column as output

#Dataset - https://raw.githubusercontent.com/CubeRind/Ukmt-hmo-floors/main/hmo.csv
#Dataset - Ukmt-hmo-floors
#Year - years
#Hamilton - minium ukmt imc score to get into hamilton maths olympiad

#1. Take the data and create dataframe
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/CubeRind/Ukmt-hmo-floors/main/hmo.csv')
#print(df)

#2. PRE PROCESSING (OPTION)
#We are not performing step no 2 here

#3. DATA VISUALISATION
#import matplotlib.pyplot as plt
#plt.scatter(x-axis,y-axis)
#plt.scatter(df['Year'],df['Hamilton'])
#plt.title('Year vs Hamilton')
#plt.xlabel('Year')
#plt.ylabel('Hamilton')

#Year - INPUT
#Hamilton - OUTPUT

#4. DIVIDE into INPUT(x) and OUTPUT(y)
#Input(x) is always a 2 dimensional array
#Output(y) is always a 1 dimensional array
#: alone in row slicing selects all rows and : alone in col slicing selects all cols
#In the column slicing part, if there is a ':', then the array is 2 dimensional
x = df.iloc[:,:1].values
#print(x) #.values covert dataframe into an array

#In the column slicing part, if there is no ':', then the array is 1 dimensional
y = df.iloc[:,1].values
#print(y)

#5. Train and Test variables
#We are not performing step no 5 here, due to less/limited data

#6. NORMALISATION/SCALING (done only for inputs)
#NORMALISATION/SCALING is done for MULTIVARIATE DATASETS
#Here our dataset is univariate

#7. MODEL BUILDING
#Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

#9. Predict the output
y_pred = model.predict(x) #Using the input values, we predict the output
#print(y_pred) #PREDICTED OUTPUT VALUES

#print(y) #ACTUAL OUTPUT VALUES

#CONCLUSION: We have to compare y_pred and y.
#When we compare y_pred and y, we come to know that there is a huge difference in the corresponding values
#This huge difference does not mean our model has predcted wrong
#It only means that our model is NOT LINEAR/LESS LINEAR
#LINEARITY of a model depends on NATURE of the data as well as SIZE of the data.

#INDIVIDUAL PREDICTION
#Now for 2000, I want to know the hamilton floor
#print(model.predict([[2000]]))

#CROSS VERIFY
#y = mx + C #Equation of a Straight line
#m - Slope
#C - y-intercept/constant
#y - dependent variable
#x - independent variable

#To find out slope(m)
m = model.coef_
#print(m)

#To find out y-intercept(C)
c = model.intercept_
#print(c)

#Now substitute m and C values in y = mx + C formula
#print(m*2000 + c)

#VISUALISATION FOR THE BEST FIT LINE
#plt.scatter(x,y) #ACTUAL VALUES
#plt.plot(x,y_pred,c = 'orange') #PREDICTED VALUES
#plt.title('BEST FIT LINE')
#plt.xlabel('Year')
#plt.ylabel('Hamilton')
#plt.show()

#R-SQUARED VALUE
#R-squared value is a statistical measure that determines the goodness of fit of a regression model
#R-squared value is always between 0 and 1
#0 means no fit
#1 means perfect fit
#R-squared value is also known as coefficient of determination
#R-squared value is calculated as 1 - (sum of squared errors)/(total sum of squares)
#Sum of squared errors is the sum of the squared differences between the predicted values and the actual values
#Total sum of squares is the sum of the squared differences between the actual values and the mean of the actual values

#To find out R-squared value
from sklearn.metrics import r2_score
r2_score(y,y_pred)

#CONCLUSION: Our model is not a good fit for the data.
