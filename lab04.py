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

# Задание 7. Возьмем в качестве категориального признака год, по факту случайны вместо монетки

sns.scatterplot(data=df,x="Growth",y="Weight",hue="Year")
plt.show()

# Задание 8 Возьмите в качестве категориального признака факультет, подойдёт вместо животных Animal

sns.scatterplot(data=df,x="Growth",y="Weight",hue="Faculty")
plt.show()

# Задание 9. Визализировать количество респондентов по полу
sns.countplot(data=df,x="Sex")
plt.show()

# Задание 10. Допустим необходимо оценить отношение мужчин и женщин в разрезе отношения к животным.
sns.countplot(data=df,x="Sex",hue="Animal")
plt.show()

# Задание 11. Поменяйте местам пол и животных, выведите на экран график.
sns.countplot(data=df,hue="Sex",x="Animal")
plt.show()

# Задание 12. Создадим новую таблицу df_Nue, в которой оставим 4 признака
df_nue =  pd.read_csv('students_nue.csv')

# Задание 13. Построить композицию графиков с использованием функции sns.pairplot(df_Nue)
sns.pairplot(data=df_nue)
plt.show()
