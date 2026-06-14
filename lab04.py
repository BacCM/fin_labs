import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('students.csv')
sns.set_theme(style="darkgrid")

# Задание 1. Построить гистограмму признака возраст
sns.displot(data=df, x="Age")
plt.show()
