def hello_1(greeting,name):
    print('%s,%s!' % (greeting,name))
def hello_2(name,greeting):
    print('%s,%s!' % (name,greeting))

hello_1('hello','world')
hello_2('hello','world')

hello_1(greeting='Hello',name='world')
hello_1(name='world',greeting='Hello')

def hello_3(name='sunliang',greeting='hello'):
    print('%s,%s!' % (name,greeting))
hello_3()
hello_3('songge')
hello_3('world')
hello_3(name='songge')
hello_3(greeting='greet')

def print_params(name,*params,**posparams):
    print(name)
    print(params)
    print(posparams)

print_params('sunliang',1,2,3,x=1,y=2,z=3)

def add(x,y): return x + y

params = (1,2)
print(add(*params))

songge = {'name':'songge','greeting':'greet'}
hello_3(**songge)
