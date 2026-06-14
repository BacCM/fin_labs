import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('students.csv')
sns.set_theme(style="darkgrid")

# Задание 2.14. Оценить исходный датафрейм df на наличие выбросов по отдельным параметрам
sns.boxplot(x=df["Growth"])
plt.show()