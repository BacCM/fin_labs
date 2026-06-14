import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('students.csv')
sns.set_theme(style="darkgrid")

# Задание 1. Построить гистограмму признака возраст
sns.displot(data=df, x="Age")
plt.show()

# Задание 2. Построить гистограмму признака вес.

sns.displot(data=df, x="Weight")
plt.show()

# Задание 3. Построить линейчатую гистограмму признака ввозраст.
sns.displot(data=df, x="Age", kind="kde")
plt.show()

# Задание 4. Построить линейчатую гистограмму признака вес.

sns.displot(data=df, x="Weight", kind="kde")
plt.show()

# Задание 5. Построить двумерный график рассеивания признаков вес и рост
sns.scatterplot(data=df,x="Growth",y="Weight")
plt.show()

# Задание 6. Для лучшей интерпретируемости можно добавить категориальный признак

sns.scatterplot(data=df,x="Growth",y="Weight",hue="Sex")
plt.show()