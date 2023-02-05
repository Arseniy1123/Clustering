from sklearn import datasets # Для работы с наборами данных 
import matplotlib.pyplot as plt # Для построения графиков
from sklearn.cluster import DBSCAN # Для использования метода DBSCAN (отчет)
from sklearn.cluster import KMeans # Для использования метода K-means
import pandas as pd
      
# Загрузка данных (для кластеризации)
irises = datasets.load_iris()

# Сорта ириса (классы)
print(irises.target_names)

# Разделение на координаты х и y 
x = irises.data[:, 0] # x - Sepal length
y = irises.data[:, 1] # y - Sepal width

# Подготовка к разделению на 3 кластера (т.к. 3 сорта) с помощью метода K-means
model = KMeans(n_clusters = 3)

# Подготовка набора данных к предсказанию и выполнение предсказания на наборе данных
model.fit(irises.data)
predictions = model.predict(irises.data)

# Построение матрицы размера 3х3 для вычисления Purity (одна из внешних мер проверки качества)
# Эта мера будет описана в отчете
ans = []
for i in range(3):
    mas_help_1 = []
    mas = []
    for k in range(len(irises.target)):
        if irises.target[k] == i:
            mas_help_1.append(k)
    for j in range(3):
        mas_help_2 = []
        for k in range(len(predictions)):
            if predictions[k] == j:
                mas_help_2.append(k)
        mas.append(len(set(mas_help_1) & set(mas_help_2)))
    ans.append(mas)
print(ans) # Вывод получившейся матрицы

# Использование внешней меры оценки качества "Purity" (формула описана в отчете)
purity = 0
for i in range(3):
    purity += max(ans[i])
purity = purity / len(irises.target)
print("Внешняя мера оценки качества 'Purity' равна", purity)

# Использование внешней меры оценки качества "Индекс Гудмэна-Крускала" (формула описана в отчете)
gk = 0
for i in range(3):
    q = sum(ans[i])
    q = q - max(ans[i])
    gk += q 
gk = gk / len(irises.target)
print("Внешняя мера оценки качества 'Индекс Гудмэна-Крускала' равна", gk)

# Построение графика по изначальному набору данных
plt.xlabel(irises.feature_names[0])
plt.ylabel(irises.feature_names[1])
plt.title('Исходный набор данных', size = 15)
plt.scatter(x, y, c = irises.target)
plt.show()

# Построение графика по предсказанному c помощью K-means набору данных 
plt.xlabel(irises.feature_names[0])
plt.ylabel(irises.feature_names[1])
plt.scatter(x, y, c = predictions)
plt.title('Предсказанный набор данных с помощью K-means', size = 15)
plt.show()

# Построение графика по предсказанному c помощью DBSCAN набору данных
# Используется метрика "Евклидово расстояние"
clusters = DBSCAN(metric = "euclidean")
clust = clusters.fit_predict(irises.data)

plt.xlabel(irises.feature_names[0])
plt.ylabel(irises.feature_names[1])
plt.scatter(x, y, c = clust)
plt.title('Предсказанный набор данных с помощью DBSCAN (Eвклидово расстояние)', size = 15)
plt.show()

# Построение графика по предсказанному c помощью DBSCAN набору данных
# Используется метрика "Расстояние Хэмминга"
clusters = DBSCAN(metric = "hamming")
clust = clusters.fit_predict(irises.data)

plt.xlabel(irises.feature_names[0])
plt.ylabel(irises.feature_names[1])
plt.scatter(x, y, c = clust)
plt.title('Предсказанный набор данных с помощью DBSCAN (расстояние Хэмминга)', size = 15)
plt.show()
