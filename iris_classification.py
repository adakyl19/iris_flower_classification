#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
df = pd.read_csv('Iris.csv')
df.drop(columns=['Id'], inplace=True)
#with some data exploration and visualization
print("=" * 60)
print("STEP 1: DATA EXPLORATION")
print("=" * 60)
print(f"\n• Shape  : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"• Columns: {list(df.columns)}")
print("\n• First 5 rows:")
print(df.head())
print("\n• Class distribution (how many flowers per species):")
print(df["Species"].value_counts())
print("\n• Any missing values?", df.isnull().sum().sum(), "← 0 means none ")
print("\n• Basic statistics (min, max, mean, std):")
print(df.describe().round(2))