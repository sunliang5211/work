def flatten(nested):
    try:
        try:nested + ''
        except TypeError: pass
        else:raise TypeError
        for sublist in nested:
            for element in flatten(sublist):
                yield element
    except TypeError:
        yield nested

l = list(flatten(['foo',['bar',['baz']]]))
print(l)
