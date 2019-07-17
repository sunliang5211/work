def faulty():
    raise Exception('Something is wrong')

def ignore_exception():
    faulty()

def handle_exception():
    try:
        faulty()
    except:
        print('Exception handled')

def describePerson(person):
    print('Description of ',person['name'])
    print('age:',person['age'])
    try:
        print('Occupation: ' + person['occupation'])
    except KeyError:pass

try:
    obj.write
except AttributeError:
    print('The object is not writeable')
else:
    print('The object is writeable')
    
