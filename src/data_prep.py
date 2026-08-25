#%% import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#%% phase1

data_path = "data/creditcard.csv"
df = pd.read_csv(data_path)
print(f"dataset shape: {df.shape}")


#%%

print("\nlast 5 rows:")
print(df.tail())
print("\ndataset information:")
df.info()
print(f"\nnumber of samples: {df.shape[0]}")
print(f"number of features: {df.shape[1] - 1}")


#%%

print("\ndescriptive statistics:")
print(df.describe())


#%%

class_counts = df["Class"].value_counts()
class_percentages = df["Class"].value_counts(normalize=True) * 100
print("\nclass describe:")
print(class_counts)
print("\nclass describe (%):")
print(class_percentages)


#%% phase2

missing_count = df.isnull().sum()
print(f"total missing values: {missing_count.sum()}")


#%% 

duplicate_rows = df[df.duplicated()]
duplicate_count = len(duplicate_rows)
print(f"total duplicate rows: {duplicate_count}")
print("\nduplicate rows by class:")
print(duplicate_rows["Class"].value_counts())


#%% 

shape_before = df.shape
df = df.drop_duplicates().reset_index(drop=True)
print(f"shape before removing duplicates: {shape_before}")
print(f"shape after removing duplicates: {df.shape}")
print("\nclass distribution after removing duplicates:")
print(df["Class"].value_counts())


#%% train test split

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


#%% 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#%% 
def prepare_data(data_path="data/creditcard.csv"):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    )
