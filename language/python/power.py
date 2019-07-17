def power(n,m):
    if m == 0:
        return 1
    else:
        return n * power(n,m-1)

print(power(2,10))
