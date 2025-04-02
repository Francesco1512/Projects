

def show_vec(V, fmt="%10.4f"):
    for x in V: print(fmt %x, end = " ")
    print()

def show_mat(M, fmt="%10.4f", head="", tail=""):
    print(head)
    for row in M:
        show_vec(row, fmt=fmt)
    print(tail)
# ------------------------------------
