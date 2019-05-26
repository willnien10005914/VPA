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



def Derivative_f(x, n, h=0.01):
    if (n == 0):
        ret = f(x)
    else:
        ret = (Derivative_f(x + h, n - 1) - Derivative_f(x, n - 1)) / h
    return ret



def Derivative_g(x, n, h=0.01):
    if (n == 0):
        ret = g(x)
    else:
        ret = (Derivative_g(x + h / 2, n - 1) - Derivative_g(x - h / 2, n - 1 )) / h

    return ret


def Square(x):
    return x * x


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


def f(x):
    """
    f(x) = 1 + x + x^2
    """
    f0 = 1
    f1 = x
    f2 = x * x
    return f0 + f1 + f2


def Taylor_Reminder(a, x, n):
    factorial = 1
    x_pow = 1

    for j in range(n + 1):
        factorial *= (j + 1)
        x_pow *= (x - a)

    # c between x and a
    c = 0.0000001
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
    return ret



"""
台北第三期
"""



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
    print("1. Create a custom function, Dericative() :")
    test_f = Square
    test_x = 3
    print("\tDerivative : %d\n" % Derivative(test_f, test_x))

    print("2. Try to create Taylor_Expansion() :")
    a = 0
    x = 3
    n = 7
    ret_g = g(x)
    ret_d = Taylor_Expansion(a, x, n) + Taylor_Reminder(a, x, n)
    err = Taylor_Reminder(a, x, n)

    print("\tg(%d) = %.2f, Taylor_Expansion(%d) = %.2f, err = %f\n" % (x, ret_g, x, ret_d, err))



if __name__ == '__main__':
    main()



