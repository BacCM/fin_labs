import pandas as pd
import numpy as np
from random import choice, randint, uniform

# Количество записей
n = 1000

# Задаем параметры для генерации (пол, возраст, рост, вес, размер обуви)
genders = ['Male', 'Female']
courses = ['Computer Science', 'Engineering', 'Mathematics', 'Physics', 'Economics', 'Biology', 'Chemistry',
           'Psychology', 'Business', 'Art']
faculties = ['Science', 'Engineering', 'Humanities', 'Business', 'Medicine']
animals_f = ['Cats', 'Cats', 'Dogs', 'Hate all']
animals_m = ['Cats', 'Dogs', 'Dogs', 'Dogs', 'Hate all']

# Списки для хранения данных
data = {
    'Student ID': [],
    'Sex': [],
    'Age': [],
    'Growth': [],
    'Weight': [],
    'Shoe size': [],
    'Course': [],
    'Faculty': [],
    'Year': [],
    'Animal': []
 }

# Генерация данных
for i in range(1, n + 1):
    sex = choice(genders)

    # Возраст: от 17 до 35 (большинство 18-25)
    age = randint(17, 35)

    # Рост: среднее 170 для Female, 178 для Male (разброс ±12 см)
    if sex == 'Male':
        height = int(np.random.normal(178, 8))
        height = max(150, min(205, height))  # ограничиваем
    else:
        height = int(np.random.normal(165, 7))
        height = max(145, min(190, height))

    # Вес: коррелирует с ростом и полом (ИМТ ~20-28)
    bmi = np.random.normal(23, 2.5)
    weight = round((bmi * (height / 100) ** 2), 1)
    weight = max(45, min(120, weight))

    # Размер обуви (EU): примерно (рост/7.5) + вариация
    if sex == 'Male':
        animal = choice(animals_m)
        shoe_base = height / 6.8
    else:
        animal = choice(animals_f)
        shoe_base = height / 7.2
    shoe_size = round(np.random.normal(shoe_base, 1.2))
    shoe_size = max(35, min(48, shoe_size))

    # Год обучения (1-4), но для старших студентов может быть 5+
    year = min(randint(1, 6), 4) if age < 24 else randint(3, 6)


    # Заполняем
    data['Student ID'].append(f"S{i:04d}")
    data['Sex'].append(sex)
    data['Age'].append(age)
    data['Growth'].append(height)
    data['Weight'].append(weight)
    data['Shoe size'].append(shoe_size)
    data['Course'].append(choice(courses))
    data['Faculty'].append(choice(faculties))
    data['Year'].append(year)
    data['Animal'].append(animal)

# Создаем DataFrame
df = pd.DataFrame(data)

# Сохраняем в CSV
df.to_csv('students.csv', index=False)

# Показываем первые 10 строк
print(df.head(10))
print(f"\nДатасет сохранен как 'students.csv' с {len(df)} записями.")
print("\nСтатистика:")
print(df.describe(include='all'))