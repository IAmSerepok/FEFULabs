using System;

class Program
{
    public delegate string CustomDelegate(Action<int> action, bool flag, char symbol);

    static void Main(string[] args)
    {
        CustomDelegate delegate1 = Method1;
        CustomDelegate delegate2 = Method2;
        CustomDelegate delegate3 = Method3;

        // Методы Action<int>
        Action<int> print1Action = (number) =>
        {
            Console.WriteLine($"Status code: {number}");
        };

        Action<int> print2Action = (number) =>
        {
            if (number == 200)
                Console.WriteLine("OK");
            else
                Console.WriteLine("ERROR");
        };

        string result1 = delegate1(print1Action, false, '7');
        Console.WriteLine($"{result1}\n");

        string result2 = delegate2(print2Action, false, 'A');
        Console.WriteLine($"{result2}\n");

        string result3 = delegate3(print1Action, true, 'C');
        Console.WriteLine($"{result3}\n");
    }

    // Метод 1
    public static string Method1(Action<int> action, bool flag, char symbol)
    {
        if (flag)
        {
            action(200);
        }
        else
        {
            action(404);
        }

        return $"Символ: {symbol}";
    }

    // Метод 2
    public static string Method2(Action<int> action, bool flag, char symbol)
    {
        int asciiCode = (int)symbol;
        action(200);

        return $"ASCII code: {asciiCode}";
    }

    // Метод 3
    public static string Method3(Action<int> action, bool flag, char symbol)
    {
        string message;

        if (!char.IsLetter(symbol))
        {
            action(404);
            return $"Ошибка: '{symbol}' не буква";
        }
        else if (char.ToUpper(symbol) == symbol)
        {
            message = "Буква в верхнем регистре";
        }
        else
        {
            message = "Буква в нижнем регистре";
        }
        if (flag)
        {
            action(200);
        }
        return message;
    }
}
