def nested_sum(l):
    total=0
    for i in l:
        if isinstance(i,list):
            total+=nested_sum(i)
        else:
            total+=i
    return total

l= [1,2,[3,4],5,7]
nested_sum(l)
