using System;
using System.Threading.Tasks;

namespace MatrixAverageTask
{
    class Program
    {
        static void Main(string[] args)
        {
            // Создаем и заполняем матрицу
            int[,] matrix = GenerateRandomMatrix(6, 8);
            
            Console.WriteLine("Сгенерированная матрица:");
            PrintMatrix(matrix);
            
            // Вычисляем среднее арифметическое с использованием Task
            double average = CalculateMatrixAverageAsync(matrix).Result;
            
            Console.WriteLine($"Среднее арифметическое элементов матрицы: {average:F2}");
        }

        // Генерация случайной матрицы
        static int[,] GenerateRandomMatrix(int rows, int cols)
        {
            Random random = new Random();
            int[,] matrix = new int[rows, cols];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = random.Next(1, 101); 
                }
            }
            
            return matrix;
        }

        // Вывод матрицы на экран
        static void PrintMatrix(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write($"{matrix[i, j],4}");
                }
                Console.WriteLine();
            }
        }

        // Синхронный метод вычисления среднего арифметического
        static double CalculateMatrixAverage(int[,] matrix)
        {           
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            long sum = 0;
            
            // Для тестирования ассинхронности
            // System.Threading.Thread.Sleep(2000);
            
            // Вычисляем сумму всех элементов
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j];
                }
            }
            
            double average = (double)sum / (rows * cols);
            
            return average;
        }

        // Асинхронный метод вычисления среднего арифметического с использованием Task
        static async Task<double> CalculateMatrixAverageAsync(int[,] matrix)
        {
            // Создаем Task для вычисления в отдельном потоке
            Task<double> calculationTask = Task.Run(() => CalculateMatrixAverage(matrix));
            
            // Ожидаем завершения задачи и получаем результат
            double result = await calculationTask;

            return result;
        }
    }
}