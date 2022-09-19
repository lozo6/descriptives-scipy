import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from statsmodels.formula.api import ols

# Creating dataframes: reading data files or converting arrays


#read csv file
data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
data

# creating arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

# Manipulating data

data.shape # 40 columns and 8 rows (40, 8)
data.columns # checks for column names
# Index(['Unnamed: 0', 'Gender', 'FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height', 'MRI_Count'], dtype='object')

print(data['Gender'])

data[data['Gender'] == 'Female']['VIQ'].mean() # simple selector

# splitting a dataframe on values of categorical variables
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

groupby_gender.mean() # checks mean or average of all columns

### EXERCISE 1 ###

# What is the mean value for VIQ for the full population?
print(data[['VIQ']].mean())

# How many males/females were included in this study?
groupby_gender = data.groupby('Gender')
print(groupby_gender.count())

# What is the average value of MRI counts expressed in log units, for males and females?
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['MRI_Count']:
    print((gender, np.log(value.mean())))

### EXERCISE 1 ###

# Plotting data

scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

### EXERCISE 2 ###

# Plot the scatter matrix for males only, and for females only. Do you think that the 2 sub-populations correspond to gender?
# scatter_matrix(data[['VIQ', 'MRI_Count', 'Height']],
#                         c=(data['Gender'] == 'Female'), marker='o',
#                         alpha=1, cmap='winter')

# fig = plt.gcf()
# fig.suptitle("blue: male, green: female", size=13)

# plt.show()

# tests if the population mean of data is likely to be equal to a given value
stats.ttest_1samp(data['VIQ'], 0)

# test if this is significant, we do a 2-sample t-test
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

# test if FISQ and PIQ are significantly different
stats.ttest_ind(data['FSIQ'], data['PIQ'])

# the variance due to inter-subject variability is confounding, and can be removed, using a “paired test”, or “repeated measures test”
stats.ttest_rel(data['FSIQ'], data['PIQ'])

# This is equivalent to a 1-sample test on the difference
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)

# T-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test, that relaxes this assumption
stats.wilcoxon(data['FSIQ'], data['PIQ'])

### EXERCISE 3 ###

# Test the difference between weights in males and females.
# male_weight = data.dropna()[data['Gender'] == 'Male']['Weight']
# female_weight = data.dropna()[data['Gender'] == 'Female']['Weight']
# stats.ttest_ind(male_weight, female_weight)

# # Use non parametric statistics to test the difference between VIQ in males and females.
# male_viq = data.dropna()[data['Gender'] == 'Male']['VIQ']
# female_viq = data.dropna()[data['Gender'] == 'Female']['VIQ']
# scipy.stats.mannwhitneyu(male_viq, female_viq)

### EXERCISE 3 ###

x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': x, 'y': y})

### EXERCISE 4 ###

# Retrieve the estimated parameters from the model above
model = ols("y ~ x", data).fit()
print(model.summary())

### EXERCISE 4 ###

data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())