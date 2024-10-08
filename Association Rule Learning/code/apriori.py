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
    confidence = rule.ordered_statistics[0].confidence
    lift = rule.ordered_statistics[0].lift
    antecedents = list(rule.ordered_statistics[0].items_base)
    consequents = list(rule.ordered_statistics[0].items_add)
    
    rule_text = f"{antecedents} -> {consequents}"
    rules_metrics.append([support, confidence, lift])
    rules_text.append(rule_text)

rules_df = pd.DataFrame(rules_metrics, columns=['support', 'confidence', 'lift'])
rules_df['rule'] = rules_text

plt.figure(figsize=(12, 8))
scatter = plt.scatter(rules_df['support'], rules_df['confidence'], c=rules_df['lift'], s = 100, cmap='Reds', alpha=0.9)

colorbar = plt.colorbar(scatter)
colorbar.set_label('Lift')

plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Scatter plot for Apriori Rules')

for i in range(len(rules_df)):
    plt.annotate(rules_df['rule'][i], 
                 (rules_df['support'][i], rules_df['confidence'][i]),
                 fontsize=8, alpha=0.7, rotation=0)

plt.grid(True)
plt.show()
