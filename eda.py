import pandas as pd
import matplotlib.pyplot as plt

filepath = 'marketing_campaign.csv'
data = pd.read_csv(filepath, delimiter='\t')

# plot histogram for columns 2-20
data.iloc[:, 2:20].hist(bins=15, figsize=(15, 10), layout=(4, 5))
plt.show()