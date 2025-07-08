import pandas as pd

file_path = "/Drugs_with_isomeric_smile.xlsx"
df = pd.read_excel(file_path)

=====

len(df)

=====

df.shape

=====

df.describe()

=====

import matplotlib.pyplot as plt

def plot_histogram(data):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plot_histogram(df["1st Replicate PrecA-gfp (GFP)"])

=====

def plot_scatter(X, Y):
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.set_xlabel("Horizonal")
    ax.set_ylabel("Vertical")

plot_scatter(df["1st Replicate PrecA-gfp (GFP)"], df["2nd Replicate PrecA-gfp (GFP)"])

=====

def plot_boxplot(data):
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels([data.name])
    ax.set_ylabel("Value")

plot_boxplot(df["1st Replicate PrecA-gfp (GFP)"])

=====

import pandas as pd

# Load the data
file_path = '/Drugs_with_isomeric_smile.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe
data.head()
