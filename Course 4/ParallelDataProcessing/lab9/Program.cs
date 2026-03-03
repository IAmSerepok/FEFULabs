using System;
using System.Linq;
using System.Threading.Tasks;

class Program
{
    static void Main(string[] args)
    {
        // Запрос размера массива у пользователя
        Console.Write("Введите размер массива: ");
        int size = int.Parse(Console.ReadLine());

        // Основная задача - генерация массива
        Task<int[]> mainTask = Task.Run(() => GenerateRandomArray(size));

        // Задачи продолжения
        Task<double> continuationTask1 = mainTask.ContinueWith(previousTask =>
        {
            int[] array = previousTask.Result;
            double result = CalculateSumOfSquaresOfEvenNumbers(array);
            Console.WriteLine($"Сумма квадратов четных элементов: {result}");
            return result;
        });

        Task<int> continuationTask2 = mainTask.ContinueWith(previousTask =>
        {
            int[] array = previousTask.Result;
            int result = FindMaxElement(array);
            Console.WriteLine($"Максимальный элемент: {result}");
            return result;
        });

        // Ожидаем завершения всех задач
        Task.WaitAll(mainTask, continuationTask1, continuationTask2);
        
        Console.WriteLine("Все задачи завершены");
    }

    // Основная задача: генерация массива случайных чисел
    static int[] GenerateRandomArray(int size)
    {
        Random random = new Random();
        int[] array = new int[size];
        
        for (int i = 0; i < size; i++)
        {
            array[i] = random.Next(1, 101); 
        }

        // Имитация задержки для тестов
        Task.Delay(500).Wait();
        
        Console.WriteLine($"Сгенерирован массив: [{string.Join(", ", array)}]");
        return array;
    }

    // Задача продолжения 1: расчет суммы квадратов четных элементов
    static double CalculateSumOfSquaresOfEvenNumbers(int[] array)
    {
        // Имитация задержки для тестов
        Task.Delay(300).Wait();

        double sumOfSquares = 0;

        for (int i = 0; i < array.Length; i++)
        {
            if (array[i] % 2 == 0)
            {
                int square = array[i] * array[i];
                sumOfSquares += square;
            }
        }

        return sumOfSquares;
    }

    // Задача продолжения 2: поиск максимального элемента
    static int FindMaxElement(int[] array)
    {
        // Имитация задержки для тестов
        Task.Delay(200).Wait();
        
        int maxElement = array.Max();
        
        return maxElement;
    }
}