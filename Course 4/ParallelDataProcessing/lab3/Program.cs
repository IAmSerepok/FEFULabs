using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    delegate int[] FindDivisibleByThreeDelegate(int arraySize);

    static async Task Main(string[] args)
    {
        // Создаем лямбда-выражение для поиска чисел, делящихся на 3
        FindDivisibleByThreeDelegate findDivisibleByThree = (arraySize) =>
        {
            // Создаем массив случайных чисел
            Random random = new Random();
            int[] numbers = new int[arraySize];

            Console.WriteLine($"Генерация массива...");

            for (int i = 0; i < arraySize; i++)
            {
                numbers[i] = random.Next(1, 1001);

                Console.Write($"{numbers[i]}, ");
            }

            Console.WriteLine("\nМассив сгенерирован.");

            // Для проверки тайм-аута
            Thread.Sleep(5000);

            // Фильтруем числа, делящиеся на 3
            int[] divisibleByThree = numbers.Where(n => n % 3 == 0).ToArray();

            return divisibleByThree;
        };

        int size = 20;
        TimeSpan timeout = TimeSpan.FromSeconds(2);

        try
        {
            // Создаем задачу для асинхронного выполнения
            var cancellationTokenSource = new CancellationTokenSource(timeout); 
            var task = Task.Run(() => findDivisibleByThree(size), cancellationTokenSource.Token);

            // Мониторинг процесса выполнения
            await MonitorTaskProgress(task, cancellationTokenSource.Token);

            // Получаем результат
            int[] result = await task;

            // Выводим результаты
            DisplayResults(result);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine($"\nТайм-аут!");
        }
    }
    
    // Метод для мониторинга прогресса выполнения задачи
    static async Task MonitorTaskProgress(Task task, CancellationToken cancellationToken)
    {              
        while (!task.IsCompleted)
        {          
            Console.Write(".");
            await Task.Delay(100, cancellationToken);
        }
    }
    
    // Метод для отображения результатов
    static void DisplayResults(int[] result)
    {
        Console.WriteLine($"\nНайдено чисел, делящихся на 3: {result.Length}");
        
        if (result.Length > 0)
        {
            Console.WriteLine("Найденные числа:");
            foreach (var number in result)
            {
                Console.Write($"{number}, ");
            }
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("Таких чисел нет");
        }
    }
}