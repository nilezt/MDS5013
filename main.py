# Suppress warnings
#import warnings

#warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd

# Load the datasets
df_012 = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
df_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df_5050 = pd.read_csv(
    'diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Display the first few rows of each dataset
df_012.head()
print("df_012 before head:", df_012.head)
df_binary.head()
df_5050.head()
