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
            
            // Фильтруем числа, делящиеся на 3
            List<int> divisibleList = new List<int>();
            for (int i = 0; i < numbers.Length; i++)
            {
                if (numbers[i] % 3 == 0)
                {
                    divisibleList.Add(numbers[i]);
                }
            }
            int[] divisibleByThree = divisibleList.ToArray();
            
            return divisibleByThree;
        };

        int size = 20;
        
        // Создаем задачу для асинхронного выполнения
        Task<int[]> task = Task.Run(() => findDivisibleByThree(size));
        
        // Мониторинг процесса выполнения
        await MonitorTaskProgress(task);
        
        // Получаем результат
        int[] result = await task;
        
        // Выводим результаты
        DisplayResults(result);
    }
    
    // Метод для мониторинга прогресса выполнения задачи
    static async Task MonitorTaskProgress(Task task)
    {       
        while (!task.IsCompleted)
        {
            Console.Write(".");
            await Task.Delay(100);
        }
    }
    
    // Метод для отображения результатов
    static void DisplayResults(int[] result)
    {
        Console.WriteLine($"Найдено чисел, делящихся на 3: {result.Length}");
        
        if (result.Length > 0)
        {
            Console.WriteLine("\nНайденные числа: ");
            for (int i = 0; i < result.Length; i++)
            {
                Console.Write($"{result[i]}, ");
            }
        }
        else
        {
            Console.WriteLine("Таких чисел нет");
        }
    }
}