import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load evaluation results
results_df = pd.read_csv('evaluation_results.csv')

# Plot episode rewards for each agent
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df)
plt.title('Agent Performance on HighwayEnv')
plt.ylabel('Episode Reward')
plt.xlabel('Agent')
plt.savefig('agent_performance.png')
plt.show()
#
