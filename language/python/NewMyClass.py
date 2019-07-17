__metaclass__=type
class NewMyClass:
    @staticmethod
    def smeth():
        print('This is a static method')


    @classmethod
    def cmeth(cls):
        print('This is a class method')

NewMyClass.smeth()
NewMyClass.cmeth()
