using System;
using System.Threading;

class Program
{
    // Делегат для метода расчета модуля вектора
    public delegate double CalculateVectorModuleDelegate(double[] vector);

    // Метод для расчета модуля вектора
    public static double CalculateVectorModule(double[] vector)
    {
        double sumOfSquares = 0;

        // Вычисляем сумму квадратов компонентов вектора
        for (int i = 0; i < vector.Length; i++)
        {
            sumOfSquares += vector[i] * vector[i];

            // Для тестирования
            Thread.Sleep(100);

            Console.WriteLine($"Поток {Thread.CurrentThread.ManagedThreadId}: обработан элемент {i + 1}/{vector.Length}");
        }

        double module = Math.Sqrt(sumOfSquares);

        Console.WriteLine($"Поток {Thread.CurrentThread.ManagedThreadId} завершил вычисление. Модуль вектора = {module:F4}");

        return module;
    }

    // Метод для генерации случайного вектора
    public static double[] GenerateRandomVector(int dimension, Random random)
    {
        double[] vector = new double[dimension];
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = random.NextDouble() * 10; 
        }
        return vector;
    }

    // Метод, который будет выполняться в каждом потоке
    public static void ThreadMethod(object parameters)
    {
        ThreadParameters threadParams = (ThreadParameters)parameters;
        double[] vector = threadParams.Vector;
        int threadNumber = threadParams.ThreadNumber;

        Console.WriteLine($"Поток {threadNumber} (ID: {Thread.CurrentThread.ManagedThreadId}) запущен");

        // Вычисляем модуль вектора
        double module = CalculateVectorModule(vector);

        Console.WriteLine($"Поток {threadNumber} (ID: {Thread.CurrentThread.ManagedThreadId}) завершен. Результат: {module:F4}");
    }

    // Класс для передачи параметров в поток
    public class ThreadParameters
    {
        public double[] Vector { get; set; }
        public int ThreadNumber { get; set; }
    }

    static void Main(string[] args)
    {
        int numberOfThreads = 3; 
        int vectorDimension = 2; 
        Random random = new Random();

        // Создаем массив потоков
        Thread[] threads = new Thread[numberOfThreads];

        // Создаем и инициализируем потоки
        for (int i = 0; i < numberOfThreads; i++)
        {
            // // Генерируем случайный вектор для текущего потока
            // double[] randomVector = GenerateRandomVector(vectorDimension, random);

            // // Создаем параметры для потока
            // ThreadParameters parameters = new ThreadParameters
            // {
            //     Vector = randomVector,
            //     ThreadNumber = i + 1
            // };

            // Создаем поток
            threads[i] = new Thread(ThreadMethod);

            Console.WriteLine($"Создан поток {i + 1} с ID: {threads[i].ManagedThreadId}");
        }

        // Запускаем все потоки
        for (int i = 0; i < numberOfThreads; i++)
        {
            threads[i].Start(GetThreadParameters(i, vectorDimension, random));
        }

        // Ожидаем завершения всех потоков
        for (int i = 0; i < numberOfThreads; i++)
        {
            threads[i].Join();
        }
    }

    // Метод для получения параметров потока
    private static ThreadParameters GetThreadParameters(int threadIndex, int dimension, Random random)
    {
        double[] randomVector = GenerateRandomVector(dimension, random);
        return new ThreadParameters
        {
            Vector = randomVector,
            ThreadNumber = threadIndex + 1
        };
    }
}
