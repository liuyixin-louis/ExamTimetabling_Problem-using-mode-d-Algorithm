


def delte_old_course_in_better_pi(tb,better_pi):
    """
    
    """
    bp = better_pi.copy()
    # 遍历better_pi中的课程
    for pi in bp.values():
        for ci in pi:
            for pj in tb:
                # 遍历tb的课程
                if ci in pj:
                    # 删除他
                    pj.remove(ci)
                    break
    delte_list = []
    for j in range(len(tb)):
        if len(tb[j])==0:
            delte_list.append(j)

    for i in sorted(delte_list,reverse=1):
        tb.pop(i)

    return tb

if __name__ == "__main__":
    tb = [[1,2,3],[4,5,6]]
    better_pi = {1:[1,2],0:[3,4]}
    result = delte_old_course_in_better_pi(tb,better_pi)
    print (result)

