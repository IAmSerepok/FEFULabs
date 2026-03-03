import numpy as np
import pandas as pd
from scipy.special import factorial


# Исходная функция
def f(x):
    return 2**2 - np.cos(x)


# Производная функции
def df(x, n):
    return - np.exp(x) / 2


# Интерполяционный полином
def L(x, points):
    res, size = 0, points.shape[0]

    for _1 in range(size):

        tmp = points['y'][_1]

        for _2 in range(size):
            if _1 != _2:
                tmp *= (x - points['x'][_2]) / (points['x'][_1] - points['x'][_2])

        res += tmp

    return res

file = open('otvet.csv', 'w')
file.write('n,delta,partial,r\n')

n_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n_list = [3 + _ for _ in range(98)]

# Формирование ответа
for n in n_list:

    # Границы отрезка
    a, b = [0.5, 1]

    # Создание df с узловыми точками
    tmp = np.linspace(a, b, n)
    points = pd.DataFrame({'x': tmp, 'y': f(tmp)})

    # Поиск погрешностей
    interval = np.linspace(a, b, 500)
    delta = np.max(np.abs(f(interval) - L(interval, points))) # Абсолютная ошибка
    partial = delta * 100 / np.max(np.abs(f(interval))) # Относительная ошибка

    # Оценка остаточного члена в форме Лагранжа
    r = np.max(np.abs(df(interval, n + 1))) * ((b - a) ** (n + 1)) / factorial(n + 1)
    print(str(n) + ',' + str(delta) + ',' + str(partial) + ',' + str(r))
    # Запись в файл
    file.write(str(n) + ',' + str(delta) + ',' + str(partial) + ',' + str(r) + '\n')
    
file.close()