from chose_best_kp_from_tb import chose_best_kp_from_tb
from delte_old_course_in_better_pi import delte_old_course_in_better_pi
from insert_better_pi_to_deltedTB import insert_better_pi_to_deltedTB
import copy

def crossover_operater(p1,p2,conflict_martix,penalty_rule,k):
    """
    input:p1,p2,conflict_martix,k,penalty_rule
    output:new_p1,new_p2
    """
    # 选出各自最好的k个时间段
    best_kp1 = chose_best_kp_from_tb(p1,penalty_rule,conflict_martix,k)
    best_kp2 = chose_best_kp_from_tb(p2,penalty_rule,conflict_martix,k)
    
    # 按照对方的k时间段进行旧表考试的删除操作
    p2 = delte_old_course_in_better_pi(p2,best_kp1)
    p1 = delte_old_course_in_better_pi(p1,best_kp2)

    # 将对方的k时间段都插入表，生成交叉之后的表
    new_p1 = insert_better_pi_to_deltedTB(p1,best_kp2)
    new_p2 = insert_better_pi_to_deltedTB(p2,best_kp1)
    
    return new_p1,new_p2
    

if __name__ == "__main__":
    k = 2
    p1 = [[0],[1],[2]]
    p2 = [[1],[2],[0]]
    penalty_rule = [4,2]
    conflict_martix = [[0,2,3],[2,0,4],[3,4,0]]
    new_p1,new_p2 = crossover_operater(p1,p2,conflict_martix,penalty_rule,k)
    print (new_p1,new_p2)