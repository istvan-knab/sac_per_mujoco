import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

end = 5000
# Reading Data
simple = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1185__train_reward.csv').head(end)
per = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1186__train_reward.csv').head(end)
ucb = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1187__train_reward.csv').head(end)

simple.columns = ["Episode", "Timestamp", "Reward"]
per.columns = ["Episode", "Timestamp", "Reward"]
ucb.columns = ["Episode", "Timestamp", "Reward"]

simple_mean = simple['Reward'].rolling(window=1).mean()
simple_std = simple['Reward'].rolling(window=1).std()

per_mean = per['Reward'].rolling(window=1).mean()
per_std = per['Reward'].rolling(window=1).std()

ucb_mean = ucb['Reward'].rolling(window=1).mean()
ucb_std = ucb['Reward'].rolling(window=1).std()

env_name = "InvertedDoublePendulum-v5"

sns.lineplot(x="Episode", y="Reward", data=simple, color='#990000', label= "Stochastic")
sns.lineplot(x="Episode", y="Reward", data=per, color='#006633', label="PER")
sns.lineplot(x="Episode", y="Reward", data=ucb, color='#003366', label="UCB")

# Adding the transparent bounds (± 1 standard deviation)
plt.fill_between(simple['Episode'], simple["Reward"] - simple_std, simple["Reward"] + simple_std, color='#990000', alpha=0.2)
plt.fill_between(per['Episode'], per["Reward"] - per_std, per["Reward"]  + per_std, color='#006633', alpha=0.2)
plt.fill_between(ucb['Episode'], ucb["Reward"]  - ucb_std, ucb["Reward"] + ucb_std, color='#003366', alpha=0.2)


ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(useOffset=False, style='plain', axis='y')

plt.legend()
plt.grid()
plt.legend(loc='lower right')
plt.show()
