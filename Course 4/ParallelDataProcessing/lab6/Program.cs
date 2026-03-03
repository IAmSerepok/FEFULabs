using System;
using System.Threading;

class Program
{
    static void Main(string[] args)
    {
        double[,] matrix = {
            { 1f, 2f, 3f, 2f, 1f },
            { 4f, 2f, 6f, 7f, 2f },
            { 2f, 9f, 1f, 2f, 4f },
            { 5f, 3f, 2f, 8f, 2f },
            { 1f, 2f, 2f, 3f, 4f }
        };

        Console.WriteLine("Матрица:");
        PrintMatrix(matrix);

        double target = 1f;

        // Используем ParameterizedThreadStart
        ParameterizedThreadStart threadStart = new ParameterizedThreadStart(SearchInMatrix);
        Thread searchThread = new Thread(threadStart);

        // Создаем объект с параметрами
        var threadData = new { Matrix = matrix, Target = target };

        searchThread.Start(threadData);
        searchThread.Join();
    }

    static void SearchInMatrix(object data)
    {
        // Извлекаем параметры
        dynamic threadData = data;
        double[,] matrix = threadData.Matrix;
        double target = threadData.Target;

        int count = 0;
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        Console.WriteLine($"Поиск элемента {target}");

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(matrix[i, j] - target) < 0.000001)
                {
                    count++;
                }
            }
        }

        Console.WriteLine($"Итого найдено вхождений: {count}");
    }

    static void PrintMatrix(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Console.Write($"{matrix[i, j],6:F2}");
            }
            Console.WriteLine();
        }
    }
}
