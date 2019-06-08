import math
import statistics
import pandas as pd
import numpy as np


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
    r_len = len(data)
    c_len = len(data[0])

    ret = [0,]*r_len
    i = 0

    for r in range(r_len):
        for c in range(c_len):
            ret[i] += data[r][c]
        i += 1


    for i in range(len(ret) - 1):
        if ret[i] != ret[i+1]:
            return False
    return True


def column_sum_check(data):
    r_len = len(data)
    c_len = len(data[0])

    ret = [0,]*c_len
    i = 0

    for c in range(c_len):
        for r in range(r_len):
            ret[i] += data[r][c]
        i += 1

    for i in range(len(ret) - 1):
        if ret[i] != ret[i+1]:
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
新竹第三期
"""
def Fibonacci(n):
    t1 = 0
    t2 = 1
    for i in range(n):
        next = t1 + t2
        t1 = t2
        t2 = next
    return t1


def reverse(data):
    i = 0
    c = 0
    for i in range(len(data) // 2):
        c = data[i]
        data[i] = data[len(data) - i - 1]
        data[len(data) - i - 1] = c
    return data


def func1(x, i, j):
    a = x[i]
    x[i] = x[j]
    x[j] = a


def func2(data):
    for i in range(len(data) - 1):
        for j in range(len(data) - 1 - i):
            if data[j] > data[j+1]:
                func1(data, j, j + 1)
    return data


def pair(data, target):
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return [i, j]
    return None


"""
台北第四期
"""
def swap(i, j, data):
    tmp = data[i]
    data[i] = data[j]
    data[j] = tmp


def reverse_1(data):
    for i in range(len(data) // 2):
        swap(i, len(data) - 1 - i, data)
    return data


def Function_3(x):
    if x > 0:
        return x
    else:
        return x / 100


def Function_4(data):
    return [Max(data), Min(data)]


def Function_5(data):
    return MinMaxScaler(data)


def MSE(y, y_hat):
    sum = 0
    for i in range(len(y) - 1):
        sum += ((y[i] - y_hat[i]) * (y[i] - y_hat[i]))

    return sum/(len(y) - 1)


def roman_numerals(num):
    values = [100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    list = []

    for i in range(len(values)):
        if (num / values[i] > 0):
            quotient = int(num / values[i])
            for j in range(quotient):
                list.append(symbols[i])
        num = num % values[i]
    return list


def RLE(list):
    rle = []
    previous = list[0]
    count = 1

    for i in range(1, len(list)):
        if previous == list[i]:
            count += 1
        else:
            rle.append(count)
            rle.append(previous)
            count = 1
            previous = list[i]

    rle.append(count)
    rle.append(previous)
    return rle


def ngram(list, n):
    ret = []
    if len(list) <= n:
        return list

    for i in range(len(list) - (n - 1)):
        tmp = []
        tmp.append(list[i])
        for j in range(1, n):
            tmp.append(list[i + j])

        ret.insert(i, tmp)
        del tmp
    return ret


def similarity(s1, s2, n):
    ns1 = ngram(s1, n)
    ns2 = ngram(s2, n)
    count = 0

    for i in range(len(ns1)):
        for j in range(len(ns2)):
            if ns1[i] == ns2[j]:
                count += 1

    return (2 * count) / (len(ns1) + len(ns2))


def Statistics(d):
    df = pd.DataFrame(d)
    #print(df['x'].value_counts(sort=False))
    #print(df.describe())
    #print("var = %.2f"%df.var())

    Q1 = np.percentile(df['x'], 25, interpolation='midpoint')
    Q3 = np.percentile(df['x'], 75)
    IQR = Q3 - Q1
    lower_inner_fence = Q1 - (1.5 * IQR)
    upper_inner_fence = Q3 + (1.5 * IQR)
    print("numpy pr 25 (Q1) : %.2f" % Q1)
    print("numpy pr IQR : %.2f" % IQR)
    print("numpy pr 75 (Q3) : %.2f" % Q3)
    print("lower_inner_fence : %.2f" % lower_inner_fence)
    print("upper_inner_fence: %.2f\n" % upper_inner_fence)

    outlier_count = 0
    for i in df['x']:
        if i > upper_inner_fence or i < lower_inner_fence:
            outlier_count += 1
            print("found outlier : %d" % i)

    print("outlier count : %d\n" % outlier_count)


    print("statistics mean : %.2f" % statistics.mean(df['x']))
    print("statistics stdev : %.2f"%statistics.stdev(df['x']))
    print("statistics median : %.2f" % statistics.median(df['x']))
    print("statistics mode : %.2f" % statistics.mode(df['x']))
    print("statistics var : %.2f" % statistics.variance(df['x']))

    if statistics.mean(df['x']) < statistics.median(df['x']) and statistics.median(df['x']) < statistics.mode(df['x']) :
        print("=> left skewed distribution")

    if statistics.mean(df['x']) > statistics.median(df['x']) and statistics.median(df['x']) > statistics.mode(df['x']) :
        print("=> right skewed distribution")

def Derangement(n):
    if (n == 1): return 0
    if (n == 0): return 1
    if (n == 2): return 1

    return (n - 1) * (Derangement(n - 1) + Derangement(n - 2))


def findStep(n):
    if (n == 1 or n == 0):
        return 1
    elif (n == 2):
        return 2
    else:
        return findStep(n - 2) + findStep(n - 1)
        """
        with step 3 option
        return findStep(n - 3) + findStep(n - 2) + findStep(n - 1)
        """


def ZscoreToX(mean, std, z):
    x = (z * std) + mean
    return x


def getPBA(PA, PB, PAB):
    intersection = PAB * PB
    return intersection / PA


def BayesRule(target, PA, PBA):
    sum = 0

    for i in range(len(PA)):
        sum += (PA[i] * PBA[i])
    return (PA[target] * PBA[target]) / sum


def P(n):
    return 1 / ((n + 1) * (n + 2))

def BayesRule_1(PA, PB):
    p_target_event = 0
    p_known_event = 0

    for i in PA:
        p_target_event += P(i)
    for i in PB:
        p_known_event += P(i)
    return p_target_event / p_known_event


def main():
    print("================程式題================================\n")
    print("%s" % "台北第一期 :")
    print("1. Compute the Formula for standard deviation :")

    test_array = [20, 3, 3, -3, -3]
    #print("\tVerify deviation by stdev : %f" % statistics.stdev(test_array))
    print("\tDeviation is '%f' in test_array%s\n" % (Deviation(test_array), test_array))

    print("2. Consider Fibonacci numbers :")
    test_index = 30
    print("\tFibonacci_recursive(%d) = %d\n" % (test_index, Fibonacci_recursive(test_index)))


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

    print("\n%s" % "新竹第三期 :")
    print("1. Please finish the Fibonacci function :")
    print("\tFibonacci(%d) = '%d'\n" % (test_index, Fibonacci(test_index)))

    print("2. Please finish the reverse function :")
    test_arr = [5, 7, 9, 1, 3, 4]
    print("\treverse(%s) =" % test_arr, end='')
    print("\t%s\n" % reverse(test_arr))

    print("3. Please write down the result of the following lines :")
    test_arr = [6, 5, 1, 8, 13, 22, 9, 1]
    print("\t%s\n" % func2(test_arr))

    print("4. Write a function which satisfies following rules :")
    test_arr = [3, 6, 2, 5, 9, 1]
    print("\t%s\n" % pair(test_arr, 10))

    print("\n%s" % "台北第四期 :")
    print("1. Given two integers i, j and a list :")
    test_arr = ['A', 'I', 'A', 'o', 'T', 'e', 'm', 'o', 'c', 'l', 'e', 'W']
    print("\treverse_1(%s) =" % test_arr)
    print("\t\t\t  %s\n" % reverse_1(test_arr))
    test_data = 5
    print("\tFunction_3(%d) = '%d'" % (test_data, Function_3(test_data)))
    test_data = -5
    print("\tFunction_3(%d) = '%.2f'" % (test_data, Function_3(test_data)))
    test_arr = [3, 16, 11, 5, 28]
    print("\tFunction_4(%s) = '%s'" % (test_arr, Function_4(test_arr)))
    test_arr = [1, 2, 3]
    print("\tFunction_5(%s) = '%s'" % (test_arr, Function_5(test_arr)))
    test_arr = [2, 4, 6, 8, 10]
    print("\tFunction_5(%s) = '%s'\n" % (test_arr, Function_5(test_arr)))
    y = [1, 2, 3, 4, 5, 6, 7]
    y_hat = [1, 2, 3, 4, 5, 6, 7]
    print("\tMSE() = '%.2f'\n" % (MSE(y, y_hat)))

    print("2-1. Roman numerals are represented y seven different symbols :")
    test_data = 88
    print("\troman_numerals(%d) =%s\n" % (test_data, roman_numerals(test_data)))

    print("2-2. Run-length encoding (RLE) :")
    test_arr = ['L', 'X', 'X', 'X', 'V', 'I', 'I']
    print("\tRLE(%s) =%s\n" % (test_arr, RLE(test_arr)))

    print("2-3. Please write a function that returns the n-gram :")
    test_arr = ['A', 'I', 'A', 'C', 'A', 'D', 'E', 'M', 'Y']
    n = 4
    print("\tn-gram(%s) =%s\n" % (test_arr, ngram(test_arr, n)))

    print("2-4. Please write a function that returns the similarity :")
    s1 = ['H', 'O', 'N', 'E', 'Y']
    s2 = ['L', 'E', 'M', 'O', 'N']
    n = 2
    print("\tsimilarity(%s, %s, %d) = %.2f\n" % (s1, s2, n, similarity(s1, s2, n)))

    print("\n================機率與統計題================================\n")

    print("上課講義 Example 7 : ")
    n = 4
    print("Derangement(%d) = %d" % (n, Derangement(n)))

    print("\n上課講義 Example 8 : ")
    n = 10
    print("findStep(%d) = %d" % (n, findStep(n)))

    print("\n台北第一期第18題 :")
    d = {
        'x': pd.Series([
            0,
            1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4,
            5, 5,
            6,
            7,
            8,
        ])
    }
    Statistics(d)

    print("\n台北第一期第11題 : ")
    PA = 0.25
    PB = 0.4
    PAB = 0.1
    print("getPBA(PA=%.2f, PB=%.2f, PAB=%.2f) = %.2f" % (PA, PB, PAB, getPBA(PA, PB, PAB)))


    print("\n台北第一期第16題 : ")
    mean = 100
    std = 15
    z = 1.2
    print("ZscoreToX(mean=%.2f, std=%.2f, z=%.2f) = %.2f" % (mean, std, z, ZscoreToX(mean, std, z)))


    print("\n台北第三期第14題 : ")
    #target for 21-30 on the index 1
    target = 1
    PA = [0.06, 0.03, 0.02, 0.04]
    PBA = [0.08, 0.15, 0.49, 0.28]
    print("BayesRule(target=%d, PA=%s, PBA=%s) = %.4f" % (target, PA, PBA, BayesRule(target, PA, PBA)))


    print("\n台北第三期第16題 : ")
    PA = [2, 3, 4]
    PB = [0, 1, 2, 3, 4]
    print("BayesRule_1(PA=%s, PB=%s) = %.4f" % (PA, PB, BayesRule_1(PA, PB)))


    print("\n台北第四期第7題 :")
    d = {
        'x': pd.Series([
            25, 60, 60, 80, 95, 100
        ])
    }
    Statistics(d)

    print("\n台北第四期第9題 :")
    d = {
        'x': pd.Series([
            83, 99, 99, 103, 103, 103,
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105,
            105, 105, 105, 105, 105, 105,
            110, 110, 110,
            110, 110, 110,
            110, 110, 110,
            113, 113, 113, 113, 113
        ])
    }
    Statistics(d)

if __name__ == '__main__':
    main()



__author__ = "Will Nien"
__email__ = "will.nien@quantatw.com"
__version__ = "1.0.2"