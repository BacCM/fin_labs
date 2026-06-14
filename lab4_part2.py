import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('students.csv')
sns.set_theme(style="darkgrid")

# Задание 2.14. Оценить исходный датафрейм df на наличие выбросов по отдельным параметрам
sns.boxplot(x=df["Growth"])
plt.show()

# Задание 14.2 - оценить параметр вес
sns.boxplot(y=df["Weight"])
plt.show()

# Задание 14.3 - оценить параметр возраст самостоятельно
sns.boxplot(x=df["Age"])
plt.show()

# 15.1.Удалим аномальные значения параметра вес.
m=df['Weight'].mean()#Вычислить среднее значения (матожидание)
print(m) # Вывести на экран значение матожидания
s=df['Weight'].std() # Вычислить среднее квадратическое отклонение
print(s) # Вывест на экра значение СКО
print(m-3*s,m+3*s) # Вычислить значение искомого интервала

df=df[(df['Weight'] > m-3*s) & (df['Weight'] < m+3*s)]
sns.boxplot(y=df["Weight"])
plt.show()