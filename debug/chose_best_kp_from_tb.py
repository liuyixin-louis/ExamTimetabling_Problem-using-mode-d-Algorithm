import copy
def chose_best_kp_from_tb(tb,penalty,conflict_martix,k):
    """
    input:tb,penalty_rule,conflict_martix,k
    output:best_k_pi_list
    """
    tb = copy.deepcopy(tb)
    best_k_pi_list = []
    penalty_tb_pi = []
    # 遍历下TB的每个时间段pi
    for i in range(len(tb)):
        peni = 0
        for ci in tb[i]:
            for j in range(len(tb)):
                if i != j: 
                #对于pi，遍历其他时间段，计算罚值总和
                    for cj in tb[j]:
                        # print (abs(i-j)-1)
                        peni += penalty[(abs(i-j)-1)]*conflict_martix[i][j]
        #记入进列表中
        penalty_tb_pi.append(peni)
    # #返回列表
    # return penalty_tb_pi

    # 关联排序并返回最优的pi时间段（k个）列表
    index = [i for i in range(len(penalty_tb_pi))]
    z = zip(penalty_tb_pi,index)
    Z = sorted(z)  # 进行逆序排列
    B, A = zip(*Z)  # 进行解压，其中的AB已经按照频率排好
    for i,j in zip(B, A):
        print(i,'\t',j)
    
    result = {}
    for i in A[:k]:
        result[i] = tb[i]
    return result
        

if __name__ == "__main__":
    k=2
    tb = [[0],[1],[2]]
    penalty_rule = [4,2]
    conflict_martix = [[0,2,3],[2,0,4],[3,4,0]]
    result = chose_best_kp_from_tb(tb,penalty_rule,conflict_martix,k)
    print (result)