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

df_w=df[(df['Weight'] > m-3*s) & (df['Weight'] < m+3*s)]
sns.boxplot(y=df_w["Weight"])
plt.show()

# 15.2. Удалите аномальные значения параметра возраст,
# выполнив аналогичные предыдущему заданию действия.

m=df['Age'].mean()#Вычислить среднее значения (матожидание)
print(m) # Вывести на экран значение матожидания
s=df['Age'].std() # Вычислить среднее квадратическое отклонение
print(s) # Вывест на экра значение СКО
print(m-3*s,m+3*s) # Вычислить значение искомого интервала

df_a=df[(df['Age'] > m-3*s) & (df['Age'] < m+3*s)]
sns.boxplot(y=df_a["Age"])
plt.show()

# Задание 16.Удалим объекты, имеющие аномальные значения исследованных выше параметров.
# Задание 16.1. Удалим аномальные значения параметра вес.
# Выведем на экран значения искомых перцинтилей
print('quantile:')
print (df["Weight"].quantile(.25),df["Weight"].quantile(.75))
a = df["Weight"].quantile(.25)
b = df["Weight"].quantile(.75)

df_w=df[(df['Weight'] > a-1.5*(b-a)) & (df['Weight'] < b+1.5*(b-a))]
sns.boxplot(x=df_w['Weight'])
plt.show()

#Задание 16.2. Удалите аномальные значения параметра возраст,
#выполнив аналогичные предыдущему заданию действия.

print('quantile age:')
print (df["Age"].quantile(.25),df["Age"].quantile(.75))
a = df["Age"].quantile(.25)
b = df["Age"].quantile(.75)

df_a=df[(df['Age'] > a-1.5*(b-a)) & (df['Age'] < b+1.5*(b-a))]
sns.boxplot(x=df_a['Age'])
plt.show()
