import xgboost as xgb


dpredict = xgb.DMatrix('predict-old-user.txt')
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('0001.model')  # load data
ypred = bst.predict(dpredict)
with open("predict-result.txt","a") as f1:
    with open("predict-old-user.txt") as f2:
        label = {'0':'4356', '1':'4355', '2':'4354', '3':'4359', '4':'3908', '5':'3909', '6':'4357', '7':'4358', '8':'4410', '9':'4625', '10':'4626', '11':'4627', '12':'4624', '13':'4677', '14':'4623', '15':'4628', '16':'4622', '17':'4657', '18':'4676', '19':'4678', '20':'3893', '21':'3885', '22':'4485', '23':'4341', '24':'3895', '25':'3886', '26':'3912', '27':'4543'}
        a = 0
        for line in f2:
            if len(line) > 1:
                b = line.split(" ")
                f1.write(b[0] + " " + 'BCAG' + label[str(ypred[a])[:-2]] + "\r")     
                a += 1
        print (a)
