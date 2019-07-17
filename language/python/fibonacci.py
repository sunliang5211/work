def calcFibs(num):
    'Calculation Fibonaccis!'
    fibs = [0,1]
    for i in range(int(num)-2):
        fibs.append(fibs[-2]+fibs[-1])
    return fibs


num = input('How many Fibonacci numbers do you want?')
print(calcFibs.__doc__)
print(calcFibs(num))
