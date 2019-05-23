import math
import statistics

def derivative(f, x, h=0.01):
    return (f(x+h) - f(x)) / h


def square(x):
    return x*x


def Fibonacci(n):

    if (n == 1):
        return 0
    elif (n == 2):
        return 1
    else:
        return Fibonacci(n-1)+Fibonacci(n-2)



def deviation(X):
    N = 0
    sum = 0
    avg = 0
    result = 0

    for i in X:
        N+=1
        sum+=i


    avg = sum/N

    for i in X:
        result+=(i-avg) * (i-avg)

    result = result/(N - 1)
    return math.sqrt(result)


def main():
    A = [20,3,3,-3,-3]
    print("derivative : %d" % derivative(square, 2))
    print("deviation : %f" % statistics.stdev(A))
    print("deviation : %f" % deviation(A))
    print(Fibonacci(30))

if __name__ == '__main__':
    main()



