using System;
using System.Collections.Generic;
using System.Threading;

// Класс элемента коллекции
public class MassItem
{
    public int[] OriginalArray;
    public List<int> EvenNumbers;
}

// Класс массива
public class MassData
{
    private List<MassItem> massive;

    public MassData()
    {
        massive = new List<MassItem>(1000);
    }

    public void Add(MassItem item)
    {
        massive.Add(item);
        ThreadPool.QueueUserWorkItem(ProcessItem, item);
    }

    private void ProcessItem(object obj)
    {
        MassItem item = (MassItem)obj;

        Console.WriteLine($"Начата обработка элемента {item.OriginalArray} в потоке {Thread.CurrentThread.ManagedThreadId}");

        // Искусственная задержка
        Thread.Sleep(1000);

        // Находим четные числа в массиве
        List<int> evenNumbers = new List<int>();
        foreach (int number in item.OriginalArray)
        {
            if (number % 2 == 0)
            {
                evenNumbers.Add(number);
            }
        }
        item.EvenNumbers = evenNumbers;

        Console.WriteLine($"Поток {Thread.CurrentThread.ManagedThreadId}: найдено {evenNumbers.Count} четных чисел");
        if (evenNumbers.Count > 0)
        {
            Console.WriteLine($"Поток {Thread.CurrentThread.ManagedThreadId}: [{string.Join(", ", evenNumbers)}]");
        }
    }
}

// Класс основной программы
class Program
{
    static void Main(string[] args)
    {
        int MaxWorkThreads, MaxIOThreads;
        ThreadPool.GetMaxThreads(out MaxWorkThreads, out MaxIOThreads);

        MassData myData = new MassData();

        Random rnd = new Random();

        for (int i = 0; i < 200; i++) // Количество уменьшено для тестов
        {
            // Создаем массив случайного размера
            int arraySize = rnd.Next(3, 16);
            int[] randomArray = new int[arraySize];

            // Заполняем массив случайными числами
            for (int j = 0; j < arraySize; j++)
            {
                randomArray[j] = rnd.Next(1, 51);
            }

            myData.Add(new MassItem() { OriginalArray = randomArray });
        }
        Console.ReadKey();
    }
}
