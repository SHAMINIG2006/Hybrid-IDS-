import pandas as pd

df = pd.read_csv("dataset.csv")

# remove class label
X = df.drop("class", axis=1)

feature_means = X.mean().to_dict()

print(feature_means)