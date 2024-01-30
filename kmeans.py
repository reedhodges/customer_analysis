import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from data_processing import process_file

filepath = 'marketing_campaign.csv'
processed_data = process_file(filepath, split_Q=False)

selected_num_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
# the categorical columns are those that start with 'Education_' and 'Marital_Status_'
selected_cat_features = [col for col in processed_data.columns if col.startswith('Education_') or col.startswith('Marital_Status_')]
selected_features = selected_num_features + selected_cat_features

df_selected = processed_data[selected_features]

# determine the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df_selected)
    inertia.append(kmeans.inertia_)

# plot elbow graph
#plt.plot(range(1, 11), inertia)
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(df_selected)
processed_data['Cluster'] = clusters

# print the mean of each feature for each cluster
print(processed_data[selected_num_features + ['Cluster']].groupby('Cluster').mean())
print(processed_data[selected_cat_features + ['Cluster']].groupby('Cluster').mean())

# plot the clusters
plt.scatter(processed_data['MntWines'], processed_data['Income'], c=processed_data['Cluster'])
plt.xlabel('MntWines')
plt.ylabel('Income')
plt.show()
