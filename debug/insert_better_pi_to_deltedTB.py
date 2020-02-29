def insert_better_pi_to_deltedTB(TB,dic_pi):
    """pi塞入修改过的tb中"""
    # 遍历dic_pi，将之放入TB中
    
    for index,pi in dic_pi.items():
        if len(TB)<=index:
            TB.append(pi)
        else:
            TB.insert(index,pi)
    return TB

if __name__ == "__main__":
    tb = [[5,6]]
    better_pi = [[1,2],[3,4]]
    dic_pi = {0:[3,4],1:[1,2]}
    result = insert_better_pi_to_deltedTB(tb,dic_pi)
    print (result)
