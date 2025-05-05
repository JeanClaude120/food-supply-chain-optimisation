#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Data Generation & Setup

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# Generate 200 warehouse IDs
warehouse_ids = [f"WH{str(i).zfill(3)}" for i in range(1, 201)]

# Generate 200 warehouse IDs
warehouse_ids = [f"WH{str(i).zfill(3)}" for i in range(1, 201)]

# Sample UK cities for locations
locations = ['London', 'Birmingham', 'Manchester', 'Glasgow', 'Leeds',
             'Bristol', 'Liverpool', 'Sheffield', 'Nottingham', 'Edinburgh']

# Simulate data
data = {
    "Warehouse_ID": warehouse_ids,
    "Location": [random.choice(locations) for _ in range(200)],
    "Storage_Capacity_Tons": np.random.randint(80, 150, size=200),
    "Daily_Intake_Tons": np.random.randint(40, 100, size=200),
    "Daily_Dispatch_Tons": np.random.randint(38, 98, size=200),
    "Avg_Storage_Duration_Days": np.round(np.random.uniform(2.0, 5.0, size=200), 1),
    "Spoilage_Rate_%": np.round(np.random.uniform(1.0, 7.0, size=200), 1),
    "Power_Outage_Hours": np.random.randint(0, 10, size=200),
    "Distance_to_Market_km": np.random.randint(10, 100, size=200),
    "On_Time_Dispatch_%": np.random.randint(75, 100, size=200),
    "Monthly_Waste_Tons": np.round(np.random.uniform(0.5, 5.0, size=200), 2),
    "Labour_Costs_£": np.random.randint(10000, 18000, size=200),
    "Transportation_Costs_£": np.random.randint(4000, 9000, size=200),
    "Temperature_Control_Issues": np.random.randint(0, 5, size=200)
}

# Create DataFrame
df = pd.DataFrame(data)

# Preview
df.head()


# In[9]:


# Exploratory Data Analysis
# Overview & Summary Statistics
df.describe()


# In[ ]:


# Correlation Heatmap


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


# Key Metrics Pairplot


# In[15]:


sns.pairplot(df[["Spoilage_Rate_%", "Daily_Dispatch_Tons", "Monthly_Waste_Tons", "Labour_Costs_£"]])
plt.show()


# In[ ]:


#Spoilage by Location


# In[17]:


plt.figure(figsize=(14, 6))
sns.boxplot(x="Location", y="Spoilage_Rate_%", data=df)
plt.xticks(rotation=45)
plt.title("Spoilage Rate by Location")
plt.show()


# In[21]:


# Clustering Warehouses Based on Performance
# Standardization & Elbow Method
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_features = df[[
    "Storage_Capacity_Tons", "Daily_Intake_Tons", "Daily_Dispatch_Tons",
    "Spoilage_Rate_%", "On_Time_Dispatch_%", "Monthly_Waste_Tons"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

# Elbow method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()


# In[ ]:


# Apply KMeans & Visualize with PCA


# In[23]:


from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA Reduction
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=df['Cluster'], palette="tab10")
plt.title("Warehouse Clusters (PCA Reduced)")
plt.show()


# In[25]:


# Regression Model to Predict Spoilage
# Build & Evaluate Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

features = [
    "Daily_Intake_Tons", "Daily_Dispatch_Tons", "Power_Outage_Hours",
    "Distance_to_Market_km", "On_Time_Dispatch_%", "Monthly_Waste_Tons",
    "Labour_Costs_£", "Transportation_Costs_£", "Temperature_Control_Issues"
]

X = df[features]
y = df["Spoilage_Rate_%"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")


# In[ ]:


# Visual Insights for Logistics Optimization
# On-Time Dispatch vs Spoilage


# In[27]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x="On_Time_Dispatch_%", y="Spoilage_Rate_%", hue="Location", data=df)
plt.title("On-Time Dispatch vs. Spoilage Rate")
plt.show()


# In[ ]:


📍 5. Visual Insights for Logistics Optimization
📦 On-Time Dispatch vs Spoilage
python
Copy
Edit
plt.figure(figsize=(8, 6))
sns.scatterplot(x="On_Time_Dispatch_%", y="Spoilage_Rate_%", hue="Location", data=df)
plt.title("On-Time Dispatch vs. Spoilage Rate")
plt.show()
🌍 Spoilage by City
python
Copy
Edit
city_spoilage = df.groupby("Location")["Spoilage_Rate_%"].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=city_spoilage.index, y=city_spoilage.values)
plt.xticks(rotation=45)
plt.title("Average Spoilage Rate by City")
plt.ylabel("Spoilage Rate (%)")
plt.show()
🚛 Transportation Cost vs Waste
python
Copy
Edit
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Transportation_Costs_£", y="Monthly_Waste_Tons", data=df)
plt.title("Transportation Cost vs Monthly Waste")
plt.show()

