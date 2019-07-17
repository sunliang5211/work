class MemberCounter:
    members = 0
    def init(self):
        MemberCounter.members += 1

m1 = MemberCounter()
m1.init()
print(MemberCounter.members)
m2 = MemberCounter()
m2.init()
print(MemberCounter.members)

print(m1.members)
print(m2.members)

m1.members = "two"
print(m1.members)
print(m2.members)

print(MemberCounter.members)

class Filter:
    def init(self):
        self.blocked = []
    def filter(self,sequence):
        return [x for x in sequence if x not in self.blocked]

class SPAMFilter(Filter):
    def init(self):
        self.blocked = ['SPAM']

f = Filter()
f.init()
print(f.filter([1,2,3]))

s = SPAMFilter()
s.init()
print(s.filter(['SPAM','SPAM','SPAM','eggs','bacon','SPAM']))

class Calculator:
    def calculate(self,expression):
        self.value = eval(expression)

class Talker:
    def talk(self):
        print('Hi,my value is ',self.value)

class talkingCalculator(Calculator,Talker):
    pass

tc = talkingCalculator()
tc.calculate('1+2+3')
tc.talk()
