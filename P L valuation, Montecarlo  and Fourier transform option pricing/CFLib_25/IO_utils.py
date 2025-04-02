
def array_show(v, wd=12, fmt="%10.4f"):
    LEN = len(v)
    nr  = LEN//wd

    off = 0
    for n in range(nr):
        x = v[off:]
        for a in x[0:wd]: print(fmt %a, end=",  ")
        print()
        off += wd
    rem = LEN%wd
    if rem > 0:
        x = v[off:]
        for a in x[0:rem]: print(fmt %a, end=",  ")
        print()
#--------------------------------
