import math
import statistics


"""
台北第一期
"""
def Deviation(X):
    N = 0
    sum = 0
    avg = 0
    result = 0

    for i in X:
        N += 1
        sum += i
    avg = sum / N

    for i in X:
        result += (i - avg) * (i - avg)

    result = result / (N - 1)
    return math.sqrt(result)


def Fibonacci_recursive(n):
    if (n < 2):
        return n
    else:
        return Fibonacci_recursive(n - 1) + Fibonacci_recursive(n - 2)


def Fibonacci(n):
    t1 = 1
    t2 = 1
    for i in range(n):
        next = t1 + t2
        t1 = t2
        t2 = next
    return t1


"""
台北第二期
"""
def Derivative(f, x, h=0.01):
    return (f(x + h / 2) - f(x - h / 2)) / h


def Square(x):
    return x * x


def Derivative_g(x, n, h=0.01):
    if (n == 0):
        ret = g(x)
    else:
        ret = (Derivative_g(x + h / 2, n - 1) - Derivative_g(x - h / 2, n - 1 )) / h

    return ret


def g(x):
    """
    g(x) = f1 + f2
        f1 = 2^x
        f2 = 2 * x^7
    """
    f1 = 1
    f2 = 1
    for i in range(int(x)):
        f1 *= 2

    for i in range(7):
        f2 *= x
    f2 *= 2
    return f1 + f2


def Taylor_Reminder(a, x, n):
    factorial = 1
    x_pow = 1

    for j in range(n + 1):
        factorial *= (j + 1)
        x_pow *= (x - a)

    # c between x and a
    c = 0.01
    return (Derivative_g(c, n + 1) * (x_pow)) / factorial


def Taylor_Expansion(a, x, n):
    ret = g(a)

    for i in range(n):

        factorial = 1
        x_pow = 1

        for j in range(i + 1):
            factorial *= (j + 1)
            x_pow *= (x - a)

        ret += ((Derivative_g(a, i + 1) * x_pow) / factorial)
    return ret + Taylor_Reminder(a, x, n)


"""
台北第三期
"""
def MinMaxScaler(data):
    ret = []
    min = Min(data)
    max = Max(data)

    for i in data:
        ret.append((i - min) / (max - min))
    return ret


def Min(data):
    ret = data[0]
    for i in range(1, len(data)):
        if ret > data[i]:
            ret = data[i]
    return ret


def Max(data):
    ret = data[0]
    for i in range(1, len(data)):
        if ret < data[i]:
            ret = data[i]
    return ret


def PassFail(grades):
    if grades is None:
        return False

    start = 0
    end = len(grades)
    status = []

    while (start < end):
        if grades[start] >= 60:
            status.append('Pass')
        else:
            status.append('Fail')
        start += 1
    return status


def MagicSquare(data):
    sum_r = 0
    sum_c = 0
    sum_d = 0

    if row_sum_check(data) is True and column_sum_check(data) is True and diagonal_sum_check(data) is True:
        for row in range(len(data)):
            sum_r += data[0][row]
            sum_c += data[row][0]
            sum_d += data[row][row]

        if sum_r == sum_c == sum_d:
            return True
        else:
            return False
    else:
        return False



def row_sum_check(data):
    r = [0,]*3
    i = 0
    for row in data:
        for col in row:
            r[i] += col
        i += 1

    for i in range(len(r) - 1):
        if r[i] != r[i+1]:
            return False
    return True


def column_sum_check(data):
    c = [0,]*3

    for row in range(len(data)):
        for col in range(len(data[row])):
            c[col] += data[row][col]

    for i in range(len(c) - 1):
        if c[i] != c[i+1]:
            return False
    return True


def diagonal_sum_check(data):
    d = [0,]*2

    length = len(data) - 1
    i = 0

    for row in range(len(data)):
        d[0] += data[row][i]
        d[1] += data[row][length - i]
        i += 1

    for i in range(len(d) - 1):
        if d[i] != d[i+1]:
            return False
    return True

"""
新竹第二期
"""



"""
台北第四期
"""


def main():
    print("%s" % "台北第一期 :")
    print("1. Compute the Formula for standard deviation :")

    test_array = [20, 3, 3, -3, -3]
    #print("\tVerify deviation by stdev : %f" % statistics.stdev(test_array))
    print("\tDeviation is '%f' in test_array%s\n" % (Deviation(test_array), test_array))

    print("2. Consider Fibonacci numbers :")
    test_index = 30
    #print("\tFibonacci_recursive(%d) = %d\n" % (test_index, Fibonacci_recursive(test_index)))
    print("\tFibonacci(%d) = '%d'\n" % (test_index, Fibonacci(test_index)))


    print("\n%s" % "台北第二期 :")
    print("1. Create a custom function, Derivative() :")
    test_f = Square
    test_x = 3
    print("\tDerivative : %d\n" % Derivative(test_f, test_x))

    print("2. Try to create Taylor_Expansion() :")
    a = 0
    x = 3
    n = 7
    ans_g = g(x)
    ans_g_taylor = Taylor_Expansion(a, x, n)
    err = ((ans_g - ans_g_taylor) / ans_g) * 100
    print("\tg(%d) = %.2f, Taylor_Expansion(%d) = %.2f, err = %.2f%%\n" % (x, ans_g, x, ans_g_taylor, err))


    print("\n%s" % "台北第三期 :")
    print("1. Create a function and named it MinMaxScaler :")
    test_arr = [4, 9, 3, 10, 0, 2]
    print("\tMinMaxScaler : %s\n" % MinMaxScaler(test_arr))

    print("2. Debug PassFail function :")
    test_grades = [10, 60, 59, 100]
    print("\tPassFail %s : %s\n" % (test_grades, PassFail(test_grades)))

    print("3. Create a function and named it MagicSquare :")
    test_arr = [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
    print("\tMagicSquare %s : %s\n" % (test_arr, MagicSquare(test_arr)))

if __name__ == '__main__':
    main()



__author__ = "Will Nien"
__email__ = "will.nien@quantatw.com"
__version__ = "1.0.0"