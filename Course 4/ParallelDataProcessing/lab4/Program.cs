using System;
using System.Threading.Tasks;

class Program
{
    // Пользовательский делегат для обратного вызова
    public delegate void SearchCallback(bool result, double searchElement);

    static async Task Main(string[] args)
    {
        // Создаем тестовую матрицу
        double[,] matrix = {
            { 4.0, 2.0, 3.0 },
            { 4.0, 5.0, 7.0 },
            { 7.0, 8.0, 9.0 }
        };

        // Выводим матрицу
        Console.WriteLine("Матрица:");
        PrintMatrix(matrix);

        double searchElement = 4.0;

        // Создаем экземпляр делегата для обратного вызова
        SearchCallback callback = new SearchCallback(SearchCompletedCallback);

        Console.WriteLine($"Ищем элемент: {searchElement}");

        // Запускаем асинхронную операцию с обратным вызовом
        SearchElementWithCallback(matrix, searchElement, callback);

        // Основной поток продолжает работу
        for (int i = 0; i < 5; i++)
        {
            Console.WriteLine($"Основной поток: {i + 1}");
            await Task.Delay(500);
        }
    }

    // Асинхронный метод с обратным вызовом
    public static async void SearchElementWithCallback(double[,] matrix, double element, SearchCallback callback)
    {
            // Запускаем поиск в отдельной задаче
            bool result = await Task.Run(() => SearchElement(matrix, element));
            
            // Вызываем обратный вызов
            callback(result, element);
    }

    // Метод поиска элемента в матрице
    public static bool SearchElement(double[,] matrix, double element)
    {
        // Для доказательства двупоточности
        Task.Delay(1000).Wait();

        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(matrix[i, j] - element) < 0.0001)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Метод обратного вызова для отображения результата
    public static void SearchCompletedCallback(bool result, double searchElement)
    {
        if (result)
        {
            Console.WriteLine($"Обратный вызов: Элемент {searchElement} найден");
        }
        else
        {
            Console.WriteLine($"Обратный вызов: Элемент {searchElement} не найден");
        }
    }

    // Метод для вывода матрицы
    public static void PrintMatrix(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{matrix[i, j],6:F1}");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}