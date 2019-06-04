import math
import os
import random
import re
import sys


def rotLeft(a, d):
    r = []
    for i in range(len(a)):
        new = i - d
        if new < 0:
            new = len(a) - 1 - new
        r.insert(new, a[i])
    return r


def minimumSwaps(arr):
    swap = 0
    count = 0
    rightPointer = len(arr) - 1
    while count < len(arr):
        arrValue = count + 1

        if arr[count] == arrValue:
            count += 1
            continue

        while arr[rightPointer] != arrValue:
            rightPointer -= 1;

        if rightPointer != count:
            print("%d %d" % (arr[count], arr[rightPointer]))
            tmp = arr[count]
            arr[count] = arr[rightPointer]
            arr[rightPointer] = tmp
            swap += 1

        rightPointer = len(arr) - 1
        count += 1
    return swap


if __name__ == '__main__':

    d = 4
    a = [1, 2, 3, 4, 5]
    result = rotLeft(a, d)

    a = [4, 3, 1, 2]
    result = minimumSwaps(a)
    print(result)
