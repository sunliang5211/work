

def str_list_to_float_list(str):
    a = []
    for x in str:
        if x == "":
            a.append(float(0))
        else:
            a.append(float(x))
    return a

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.train_users,self.train_items  = self.load_rating_file_as_list(path + ".train")
        self.test_users, self.test_items  = self.load_rating_file_as_list(path + ".test")
        self.testNegatives = self.load_negative_file(path + ".negative")
        assert len(self.test_items) == len(self.testNegatives)
        
    def load_rating_file_as_list(self, filename):
        users = []
        items = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.split(",")
                user = str_list_to_float_list(arr[2:28])
                a = str_list_to_float_list(arr[31:36])
                user.extend(a)
                item = [float(x) for x in arr[37:]]
                users.append(user)
                items.append(item)
        
        print("Load  " + filename + "   done!")
        return users,items

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.split(",")
                flow = [float(x) for x in arr]
                neg = []
                for x in range(49):
                    neg.append(flow[55*(x+1)-54:55*(x+1)+1])
                negativeList.append(neg)
        print("Load  " + filename + "   done!")
        return negativeList



           
