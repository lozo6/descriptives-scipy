import pandas as pd
import numpy as np

#read csv file
data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
data

# creating arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

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

### EXERCISE ###

