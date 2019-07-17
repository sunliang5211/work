with open("product","r") as f1:
    with open("train.product.txt","a") as f2:
        with open("test.product.txt","a") as f3:
            line_num = 0
            num = [0]*28
            label = {'4356':0, '4355':1, '4354':2, '4359':3, '3908':4, '3909':5, '4357':6, '4358':7, '4410':8, '4625':9, '4626':10, '4627':11, '4624':12, '4677':13, '4623':14, '4628':15, '4622':16, '4657':17, '4676':18, '4678':19, '3893':20, '3885':21, '4485':22, '4341':23, '3895':24, '3886':25, '3912':26, '4543':27}
            for line in f1:
                a = line.split("|")
                line_num += 1
                if line_num%10 != 0:
                    f2.write(str(label[a[0][4:]]) + " ")
                else:
                    f3.write(str(label[a[0][4:]]) + " ")
                for i in range(len(a) -2):
                    if line_num%10 != 0:
                        f2.write(str(i+1) + ":" + a[i+2] + " ")
                    else:
                        f3.write(str(i+1) + ":" + a[i+2] + " ")
                num[label[a[0][4:]]] += 1
            print(num)        
            print(label)

