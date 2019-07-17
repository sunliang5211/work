class Calc:
    count = 0

    def print():
        print('I\'m sunliang')
    
    def printCalc(self,x):
        foo = lambda x:x*x
        print(foo(x))
        Calc.count += 1

    def printCount(self):
        print(Calc.count)

print(Calc.count)
a = Calc()
a.printCalc(2)
b = Calc()
b.printCalc(3)

print(Calc.count)
print(a.count)
print(b.count)

a.count = "sunliang"
print(Calc.count)
print(a.count)
print(b.count)
a.printCalc(2)
b.printCalc(3)
print(Calc.count)
print(a.count)
print(b.count)
Calc.print()



