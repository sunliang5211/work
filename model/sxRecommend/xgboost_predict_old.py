import xgboost as xgb


dtest = xgb.DMatrix('test.product.txt')
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('0001.model')  # load data
ypred = bst.predict(dtest)
with open("predict-result-old.txt","a") as f1:
    with open("test.product.txt") as f2:
        a = 0
        for line in f2:
            if len(line) > 1:
                f1.write(str(ypred[a])[:-2] + " " + line)        
                a += 1
        print (a)
