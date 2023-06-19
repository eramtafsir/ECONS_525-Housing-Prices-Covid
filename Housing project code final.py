# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:40:08 2023

@author: eramt
"""

#  After trying so many versions, here is the final script used for the project.

# "Impact of Covid-19 on Housing Price Dynamics in Washington" -- Project by Eram Tafsir



import pandas as pd
import os
import numpy as np
import wooldridge as woo
import statsmodels.formula.api as smf
import math
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import patsy as pt
import scipy.stats as stats
import linearmodels as plm
import seaborn as sns
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import compare
from statsmodels.iolib.summary2 import summary_col


from stargazer.stargazer import Stargazer


#Defining Variables

# county -  refers to any of the 10 selected counties in Washington state, namely - Whitman, Chelan, Okanogan, Yakima, Grant, King, Snohomish, Pierce, Thurston and Spokane
# County_Code - refers to the code assigned to each of the 10 selected counties
# date - quarterly time-series starting from Q1-2018 to Q2-2022
# county_type - Dummy variable indicating whether a particular county is rural=0 or urban=1
# Price - It is the dependent variable which refers to the Median Sale Price of House in each county in US dollars
# unemp - refers to the county-levelcounty-level unemployment rate
# inc - refers to the household income
# mor - refres to the 30-year fixed mortgage rate in US
# death - refers to the county-level covid-19 mortality rate
# cases - refres to the county-level covid-19 cases
# D1 - Time Dummy vatiable for the period post outbreak from Q1-2020
# D2 - Time Dummy vatiable for the period post introduction of Vaccination from Q4-2020
# D3 - Time Dummy vatiable for the period post Reopening with Full Capacity from Q3-2021
# t - refers to the time trend variable taking values from 1, 2, 3.... 18

# The block of code below auto-clears the console each time you run the program 
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# Auto clears plots from previous runs
f = plt.figure()
f.clear()
plt.close(f)

# The option below will force the print command to display all of the columns requested 
pd.set_option('display.max_columns', None)

plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)

# Define Path
#os.chdir('C:/Users/eramt/OneDrive/Desktop/Ecotrix project/My project/Data sets/Python Code Housing Project/Project submission')
os.chdir('C:/Users/eramt/OneDrive/Desktop/Ecotrix/Ecotrix project/My project/Data sets/Python Code Housing Project/Project submission')

                # Brief about the data:
        
# A new data set was created after retreieving all the required data from different sources.
# Data was then stored in an CSV files.
# The data set created is a panel data is that we can be used to control heterogeneity in our regression model by acknowledge heterogeneity as fix or random.
# Since, all units are tracked for the same number of time periods, the data panel is called a balanced panel.
# Since the same set of units is tracked throughout the study, it’s called a fixed panel.
# Thus, our data set is fixed and balanced.

#Errors introduced due to our inability to measure certain characteristics that are unit-specific.


                #Importing and Organising Data

#Import External Data File - Combined with all the relevant data
df1 = pd.read_csv('data1.csv',delimiter=',')

df1['date'] = pd.to_datetime(df1['date'])

# look at imported data:
print(f'df1.head(): \n{df1.head()}\n')
print(f'df1.tail(): \n{df1.tail()}\n')

# Find the summary statistics for all data 
Data_summary = df1.describe()
print(f'Data_summary:\n{Data_summary}\n')

# Find the correlation coeficients for all data 
Corr_coef = df1.corr()
print(f'Corr_coef:\n{Corr_coef}\n')

#Checking for data types:
c = df1.dtypes
print(f'{c}')

# Note that there are 10 counties which are tracked over 18 quarters starting from Q1 2018 to Q2 2022(n =10, T = 18)

N = len(df1)
print(f'Total observations: {N}\n')



                # Define Dummy Variables

#To define Counties as Rural and Urban, we will use the following criteria
# Rural = 0, Counties with a population density less than 100 persons per square mile or counties smaller than 225 square miles as of April 1, 2022
# Urban - 1, Counties with a population density greater than and equal to 100 persons per square mile or counties reater than and equal to 225 square miles as of April 1, 2022
#According to it, out of the 10 Washington counties in our data set:
    
            # Whitman, Chelan, Okanogan, Yakima and Grant are Rural Counties
            # King, Snohomish, Pierce, Thurston and Spokane are Urban Counties

df1['County_type'] = np.where(df1['County'].isin(['King','Snohomish','Pierce', 'Thurston','Spokane']), 1, 0)

# Define Dummy Variables based on Time Period
#D1 - Post Covid outbreak from Q1-2020
#D2 - Vaccination Introduction from Q4-2020
#D3 - Reopening with Full Capacity from Q3-2021

df1['D1'] = np.where(df1['date'] >= '2020-01-01', 1,0)
df1['D2'] = np.where(df1['date'] >= '2020-10-01', 1,0)
df1['D3'] = np.where(df1['date'] >= '2021-07-01', 1,0)



            # Plotting data

df3 = pd.read_csv('Transposed.csv',delimiter=',')

df1['date'] = pd.to_datetime(df1['date'])
print (df1.head())

df2 = df1['date']

df4 = pd.concat([df2, df3], axis = 1, join='inner')

T = len(df4)

# monthly time series starting Feb. 1978:
df4.index = pd.date_range(start='2018-01-01', periods=T, freq='QS')
print(f'df4.head(): \n{df4.head()}\n')


# plot chnimp (default: index on the x-axis):
plt.plot('Chelan', data=df4, color='black', linestyle='-')
plt.ylabel('Chelan')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/ChelanPrice.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Grant', data=df4, color='black', linestyle='-')
plt.ylabel('Grant')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('King county', data=df4, color='black', linestyle='-')
plt.ylabel('King County')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/King_CountyPrice.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Okanogan', data=df4, color='black', linestyle='-')
plt.ylabel('Okanogan')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Okanogan.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Pierce', data=df4, color='black', linestyle='-')
plt.ylabel('Pierce')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Snohomish', data=df4, color='black', linestyle='-')
plt.ylabel('Snohomish')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Whitman', data=df4, color='black', linestyle='-')
plt.ylabel('Whitman')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Thurston', data=df4, color='black', linestyle='-')
plt.ylabel('Thurston')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Yakima', data=df4, color='black', linestyle='-')
plt.ylabel('Yakima')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')

# plot chnimp (default: index on the x-axis):
plt.plot('Spokane', data=df4, color='black', linestyle='-')
plt.ylabel('Spokane')
plt.xlabel('time')
plt.show()
#plt.savefig('PyGraphs/Grant.pdf')


        # Setting up a panel:

data_o = df1.set_index(['County_Code', 'date'], drop=False)
summary = data_o.describe()
print(summary)



        # Plotting the trend in hosuing price in the panel data :
            
#Ploting
colors = ['blue', 'red', 'orange', 'lime', 'yellow', 'cyan', 'violet', 'green', 'pink','black' ]
 

sns.lineplot(x=data_o['date'], y=data_o['Price'],hue=data_o['County'], palette=colors).set(title='Trend in Median House Prices across Counties')
plt.show()

#Ploting
colors1 = ['blue', 'red', 'orange', 'lime', 'yellow', 'cyan', 'violet', 'green', 'pink','black' ]
 
sns.lineplot(x=data_o['date'], y=data_o['inc'],hue=data_o['County'], palette=colors1).set(title='Trend in Median House Prices across Counties')
plt.show()



        # Transforming varibales: 
            
# Since plotting shows and exponential increase in prices, log-transforming this variable converted the relationship from exponential to linear, which serves the purpose of improving metrics of the linear regression.
data_o["lprice"] = np.log(data_o["Price"])

#Plotting histograms price vs log price
plt.hist(data_o["Price"],bins=100)
plt.title("Housing Prices")
plt.show()

plt.hist(data_o["lprice"],bins=100)
plt.title("Log of Housing Prices")
plt.show()
# Thus, log price histogram is more normal.


#Plotting other histograms to check distribution of data:
plt.hist(data_o["inc"],bins=100)
plt.title("Household income")
plt.show()

#Plotting other histograms to check distribution of data:
plt.hist(data_o["unemp"],bins=100)
plt.title("unemployment rate")
plt.show()



        #Calculating VIF of independent variables :

X = data_o[['mor', 'death', 'unemp', 'cases', 'D1', 'D2', 'D3', 'inc', 'County_type']]
vif = pd.DataFrame()
vif["variable"] = X.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(f' VIF : {vif}\n')

# Due to high VIF between unemp and inc
# We will drop income
# We will also drop 'cases', to rule out any multicollinearity between cases ans deaths.

X = data_o[['mor', 'death', 'unemp', 'D1', 'D2', 'D3', 'County_type']]
vif = pd.DataFrame()
vif["variable"] = X.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(f' VIF : {vif}\n')



        # Analysing data by grouping rural vs urban counties

        # Plotting the trend in Housing Prices:

df5 = pd.read_csv('mean.csv',delimiter=',')

df5 = df5.set_index(['County_type','date'], drop=False)

colors = ['blue', 'red']
sns.lineplot(x=df5['date'], y=df5['Price'], hue=df5['County_type'], palette=colors).set(title='Trend in Median House Prices in Rural vs Urban')
plt.show()


colors1 = ['blue', 'red', 'orange', 'lime', 'yellow', 'cyan', 'violet', 'green', 'pink','black' ]
sns.lineplot(x=data_o['date'], y=data_o['inc'],hue=data_o['County'], palette=colors1).set(title='Trend in Household income across Counties')
plt.show()

colors2 = ['blue', 'red', 'orange', 'lime', 'yellow', 'cyan', 'violet', 'green', 'pink','black' ]
sns.lineplot(x=data_o['date'], y=data_o['unemp'],hue=data_o['County'], palette=colors1).set(title='Trend in unemployment rate across Counties')
plt.show()




        #Estimating different models by including time trend and its interaction with other variables:

            
    # i) Pooled OLS:
reg_ols = plm.PooledOLS.from_formula(
    formula='lprice ~ t*(D1 + D2 + D3 + mor + death + County_type + unemp) ', data=data_o)
results_ols = reg_ols.fit(cov_type ='clustered', cluster_entity = True)
print(f'results_ols: {results_ols}\n')

    # ii) Random Effects Model:
reg_re = plm.RandomEffects.from_formula(
    formula='lprice ~ t*( D1 + D2 + D3 + mor + death + unemp + County_type) ', data=data_o)
results_re = reg_re.fit(cov_type ='clustered', cluster_entity = True)
b_re = results_re.params
b_re_cov = results_re.cov
print(f'results_re: {results_re}\n')


    # iii) Fixed Effects Model:
reg_fe = plm.PanelOLS.from_formula(
    formula='lprice ~ t * (D1 + D2 + D3 + mor + death + unemp + County_type) + '
            'EntityEffects', data=data_o, drop_absorbed = True)
results_fe = reg_fe.fit(cov_type ='clustered', cluster_entity = True)
b_fe = results_fe.params
b_fe_cov = results_fe.cov
print(f'results_fe: {results_fe}\n')


# print results:
theta_hat = results_re.theta.iloc[0, 0]
print(f'theta_hat: {theta_hat}\n')

table_ols = pd.DataFrame({'b': round(results_ols.params, 4),
                          'se': round(results_ols.std_errors, 4),
                          't': round(results_ols.tstats, 4),
                          'pval': round(results_ols.pvalues, 4)})
print(f'table_ols: \n{table_ols}\n')

table_re = pd.DataFrame({'b': round(results_re.params, 4),
                         'se': round(results_re.std_errors, 4),
                         't': round(results_re.tstats, 4),
                         'pval': round(results_re.pvalues, 4)})
print(f'table_re: \n{table_re}\n')

table_fe = pd.DataFrame({'b': round(results_fe.params, 4),
                         'se': round(results_fe.std_errors, 4),
                         't': round(results_fe.tstats, 4),
                         'pval': round(results_fe.pvalues, 4)})
print(f'table_fe: \n{table_fe}\n')

        
            # Hausman test of FE vs. RE

# (I) find overlapping coefficients:
common_coef = set(results_fe.params.index).intersection(results_re.params.index)
print(f'common_coef: {common_coef}\n')
# (II) calculate differences between FE and RE:
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
data_o = len(b_diff)
b_diff.reshape((data_o, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((data_o, data_o))
# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, data_o)
print(f'stat: {stat}\n')
print(f'pval: {pval}\n')
#Hauss man test null hypothesis that given model is random effects is rejected.
#Therefore, the given model is fixed effects

#Hauss man test null hypothesis that given model is random effects is rejected.
#Therefore, we will choose model is the fixed effects model.

print(compare({"Pooled": results_ols, "RE": results_re, "FE": results_fe}))


