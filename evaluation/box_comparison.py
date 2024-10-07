import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

end = 5000
# Reading Data
simple = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1185__train_reward.csv').head(end)
per = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1186__train_reward.csv').head(end)
ucb = pd.read_csv('InvertedDoublePendulum-v5/SACPERM-1187__train_reward.csv').head(end)