with open("predict-old","r") as f1:
    with open("predict-old-user.txt","a") as f2:
        line_num = 0
        for line in f1:
            a = line.split("|")
            line_num += 1
            f2.write(str(a[0]) + " ")
            for i in range(len(a)-1):
                f2.write(str(i+1) + ":" + a[i+1])
        print(line_num)        

