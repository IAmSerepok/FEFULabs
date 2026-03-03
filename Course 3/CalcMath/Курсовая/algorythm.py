import numpy as np


def decompose_LU_hessenberg(A):
    """
    Функция, производящая LU разложение матрицы A
    """

    w, h = A.shape
    if w != h:
        raise ValueError("Матрица не квадратная")
    n = w

    L, U = np.zeros((n, n), float), np.zeros((n, n), float)

    # i = 0 Первая строка матрицы U
    for j in range(n):
        U[0, j] = A[0, j]

    # j = 0 Первый столбец матрицы L
    for i in range(n):
        L[i, 0] = A[i, 0] / U[0, 0]

    # i = j Диагональ матрицы L
    for index in range(n):
        L[index, index] = 1

    # Остальные эллементы матриц L и U
    for i in range(1, n):
        for j in range(1, min(n, i + 2)):
            if i <= j:
                U[i, j] = A[i, j]
                for k in range(i):
                    U[i, j] -= L[i, k] * U[k, j]

            else:
                L[i, j] = A[i, j]
                for k in range(j):
                    L[i, j] -= L[i, k] * U[k, j]
                L[i, j] /= U[j, j]

    return L, U


def householder_reflection(a):
    """Создает отражение Хаусхолдера для вектора a."""
    norm_a = np.linalg.norm(a)
    sign = -1 if a[0] < 0 else 1
    u1 = a[0] + sign * norm_a
    u = a.copy()
    u[0] = u1
    u[1:] = a[1:]
    u /= np.linalg.norm(u)
    return u


def hessenberg(A):
    """Преобразует произвольную матрицу A в матрицу Хессенберга с сохранением собственных значений."""
    A = A.astype(float).copy()  # создаем копию матрицы, чтобы не изменять исходную
    n = A.shape[0]
    
    for k in range(n - 2):
        # Выбираем подстолбец начиная с k+1
        x = A[k+1:n, k]
        u = householder_reflection(x)

        # Формируем матрицу отражения
        H = np.eye(n)
        H[k+1:n, k+1:n] -= 2 * np.outer(u, u)

        # Обновляем A
        A = H @ A @ H

    return A


def compose(L, U):
    """
    Оптимизированный вариант умножения матриц для получения произведения UL
    """

    n = L.shape[0]



    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n - 1):
            A[i, j] = U[i, j] + U[i, j + 1] * L[j + 1, j]

    for i in range(1, n):
        A[i, i - 1] = U[i, i] * L[i, i - 1]

    for i in range(n):
        A[i, n - 1] = U[i, n - 1]

    return A


def LU_eig(A0, eps):
    """
    LU алгоритм поиска собственных значений
    """

    A0 = hessenberg(A0)
    A = A_old = A0
    
    n = A.shape[0]

    while True:
        L, U = decompose_LU_hessenberg(A)
        A = compose(L, U)

        if np.sum(np.abs([A[i] - A_old[i] for i in range(n)])) < eps:
            break

        A_old = A
    
    return np.diag(A)


if __name__ == "__main__":
    # Пример использования
    A = np.random.rand(5, 5) * 0.1

    A = A.T @ A

    result = LU_eig(A, eps=0.01)
    print(np.round(result, 2))
