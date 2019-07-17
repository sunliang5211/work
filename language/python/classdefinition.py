__metaclass__=type #使用新式类

class Person:
    def setName(self,name):
        self.name = name
    def getName(self):
        return self.name
    def greet(self):
        print("Hello,world! i'm %s." % self.name)

foo = Person()
bar = Person()

foo.setName('Luke Skywalker')
bar.setName('Anakin Skywalker')

foo.greet()
bar.greet()

print(foo.name)
print(bar.name)

foo.name = 'Yoda'
foo.greet()

class Class:
    def method(self):
        print('I have a self!')
def function():
    print("I don't ...")

instance = Class()
instance.method()

instance.method = function
instance.method()

class Bird:
    song = 'Squaawk!'
    def sing(self):
        print(self.song)

bird = Bird()
bird.sing()
birdsong = bird.sing
birdsong()

class Secretive:
    def __inaccesible(self):
        print("Bet you can't see me")
    def _method(self):
        print("my method")
    def accessible(self):
        print("The secret message is :")
        self.__inaccesible()

s = Secretive()
s._method()
s._Secretive__inaccesible()
s.accessible()

