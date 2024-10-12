import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori

dataset = pd.read_csv('../data/Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

rules_list = list(rules)

rules_metrics = []
rules_text = []

for rule in rules_list:
    support = rule.support
    antecedents = list(rule.ordered_statistics[0].items_base)
    consequents = list(rule.ordered_statistics[0].items_add)
    
    rule_text = f"{antecedents} -> {consequents}"
    rules_metrics.append(support)
    rules_text.append(rule_text)

eclat_df = pd.DataFrame(rules_metrics, columns=['Support'])
eclat_df['Itemset'] = rules_text

eclat_df = eclat_df.sort_values(by='Support', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(eclat_df['Itemset'], eclat_df['Support'], color='skyblue')
plt.xlabel('Support')
plt.title('Top 10 Frequent Itemsets by Support')
plt.gca().invert_yaxis()
plt.show()
