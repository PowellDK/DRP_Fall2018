# -*- coding: utf-8 -*-
"""
Directed Reading Program Final Project Replication
Statistical Analysis of Midterm Congressional Losses by Sitting President's Party 1950-2014
Projection of 2018 Midterm Election

Performed by Devin K. Powell
"""
import pandas
import numpy
import matplotlib.pyplot as plt

data=[]
with open("C:\\Users\\pwlld\\Documents\\DRPReplication.csv","r") as inputfile:
    for line in inputfile:
        data.append(line.strip("\n").split(","))
frame=pandas.DataFrame(data[1:], columns=data[0])
for i, row in frame.iterrows():
    row[0]=float(row[0])
    row[1]=float(row[1])
    row[2]=float(row[2])

#Mean Calculations
sum_SG = 0.0
sum_AR = 0.0
sum_UR = 0.0
for i, row in frame.iterrows():
    sum_SG=sum_SG+float(row[0])
    sum_AR=sum_AR+float(row[1])
    sum_UR=sum_UR+float(row[2])
mean_SG=sum_SG/17.0
mean_AR=sum_AR/17.0
mean_UR=sum_UR/17.0

#1 Variable Linear Regression: Approval Rating
beta0_1Var = 0.0
beta1_1Var = 0.0
num=0.0
denom=0.0
for i, row in frame.iterrows():
    num=num+((float(row[1])-mean_AR)*(float(row[0])-mean_SG))
    denom=denom+((float(row[1])-mean_AR)*(float(row[1])-mean_AR))
beta1_1Var=num/denom
beta0_1Var=mean_SG-(beta1_1Var*mean_AR)
print ("Seat Gains=",beta0_1Var,"+",beta1_1Var,"*Approval Rating")

plt.scatter(frame['Approval Rating'], frame['Seat Gains'], color='red')
plt.title('Approval Rating vs Seat Gains', fontsize=14)
plt.xlabel('Approval Rating', fontsize=14)
plt.ylabel('Seat Gains', fontsize=14)
plt.plot([0,70], [beta0_1Var,beta0_1Var+70*beta1_1Var], '-k')
plt.grid(True)
plt.show()


#2018 Prediction: 1 Variable Linear Regression-Approval Rating
SG_2018 = beta0_1Var+(beta1_1Var*40)
print ('Prediction for 2018: ',SG_2018)

#1 Variable Linear Regression: Unemployment Rate
beta0_1VarU = 0.0
beta1_1VarU = 0.0
numU=0.0
denomU=0.0
for i, row in frame.iterrows():
    numU=numU+((float(row[2])-mean_AR)*(float(row[0])-mean_SG))
    denomU=denomU+((float(row[2])-mean_AR)*(float(row[2])-mean_AR))
beta1_1VarU=numU/denomU
beta0_1VarU=mean_SG-(beta1_1VarU*mean_AR)

print ("Seat Gains=",beta0_1VarU,"+",beta1_1VarU,"*Unemployment")
 
plt.scatter(frame['Unemployment Rate'], frame['Seat Gains'], color='green')
plt.title('Unemployment Rate vs Seat Gains', fontsize=14)
plt.xlabel('Unemployment Rate', fontsize=14)
plt.ylabel('Seat Gains', fontsize=14)
plt.plot([0,11], [beta0_1VarU,beta0_1VarU+11*beta1_1VarU], '-k')
plt.grid(True)
plt.show()

#2018 Prediction: 1 Variable Linear Regression-Unemployment Rate
SG_2018U = beta0_1VarU+(beta1_1VarU*3.7)
print ('Prediction for 2018: ',SG_2018U)

#Multiple Regression
from sklearn import linear_model
import statsmodels.api as sm

X = frame[["Approval Rating", "Unemployment Rate"]] ## X usually means our input variables (or independent variables)
y = frame["Seat Gains"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: ', regr.intercept_)
print('Coefficients: ', regr.coef_)
print('Prediction for 2018: ', regr.predict([[0, 40, 3.7]]))

print('Percent error: ', (abs((-23-regr.predict([[0, 40, 3.7]]))/-23))*100)

