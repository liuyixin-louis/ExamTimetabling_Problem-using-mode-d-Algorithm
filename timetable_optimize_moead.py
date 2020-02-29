#!/usr/bin/env python
# coding: utf-8

# In[893]:


"""
完成情况：
混合初始化算子：暂时只有随机生成，混合启发未写
交叉算子：实现，还有bug，交叉时解会丢班次
变异算子：暂时用的简单交换班次，启发重排未写
结果可视化：初步实现
"""


# In[ ]:


#导入必要的库函数
import copy,time
import numpy as np
import copy
from itertools import combinations
import matplotlib.pyplot as plt


# In[894]:


#定义进化算法的参数及数据结构
iteration=200 #迭代次数
population=20
archive=200 #存档数目
nObj=2
# nVar=int #决策变量个数
# varMax=np.ones(nVar)
# varMin=np.zeros(nVar)
penalty_par = [16,8,4,1]#罚参数
ratio=0.1 #交换比率
Pc = 0.8 #交叉概率
PM = 0.2#变异概率
PReinsert = 0.2 #重排比例
# yita1=2 #交叉参数
# yita2=5 #变异参数               
T=5#领域规模
seed_p = 617

'''下面不是参数！'''
Lambda=[] #权重向量
pops=[]
EPs=[]
np.random.seed(seed_p)

#DEBUG:
Debug_bianyi_P = []


# In[895]:


#读取数据
#学生数据：每个学生参与课程编号列表；
#课程数据：课程编号，以及对应的时间耗时

stupath = './ear-f-83-2.stu'
crspath = './ear-f-83-2.crs'
studata = []
with open(stupath) as f:
    line = f.readline()
    while line:
#         print(line.split())
        line = f.readline()
        l = line.split()
        for i in range(len(l)):
            l[i] = int(l[i])
        studata.append(l)
crsdata = np.loadtxt(crspath,dtype = int)
stu_num = len(studata)
Enum = len(crsdata)
# crsdata
# studata


# In[896]:


#工具函数：
def check_conflict_m():
    for elist in studata:
        for k in range(len(elist)-1):
            if not conflict_martix[elist[k]-1][elist[k+1]-1]:#不冲突
                #冲突矩阵出问题了！
                print ('wrong!')
                return 
    print ('may be right')
# check_conflict_m()

def calculate_penalty(delta):
    if delta>=5:
        return 0
    else:
        return  penalty_par[delta-1]
    
def check_TB_right(TB):
    #检查硬性条件
    for pi in list(TB):
        for k in range(len(pi)-1):
            if conflict_martix[pi[k]][pi[k+1]]:#冲突
                print (pi[k],pi[k+1],'is conflict')
                print ('解出问题了：安排有冲突')
                return False 
    #检查完备性及有无重复
    ls = []
    tb2Check = list(TB)[:]
    #展开并sort下
    for l in tb2Check:
        ls += l
    ls.sort()
#     print (ls)
    stand = [i for i in range(Enum)]
    if ls!=stand:
#         print (ls)
        print ('解出问题了：没全安排好,差{}门考试没安排'.format(len(stand)-len(ls)))
        return False
    #和range(crs_num)对比，完全相同则为正确，否则则错
    
#     print ('y')
    return True
    #前面都没出错则正确
    #结束

def isDominates(x,y):
    #x是否支配y
#     print (x)
#     print (y)
#     print (x<=y)
    try:
        if x<y:
            return True
        elif x>y:
            return False
    except:
        return (x<=y).all() and (x<y).any()
def determinDomination(p):
    '''决定支配关系'''
    for i in range(len(p)):
        p[i].dominate=False
    for i in range(0,len(pops)-1):
        for j in range(i+1,len(p)):
            if isDominates(p[i].cost,p[j].cost):
                p[j].dominate=True#j被i支配
            if isDominates(p[j].cost,p[i].cost):
                p[i].dominate=True
def delte_cd_in_p(cd,p):
    '''从表p中删去cd课程'''
    for pi in p:
        if cd in pi:
            pi.remove(cd)
    for pi in p:
        if pi is []:
            p.remove(pi)
    return p


# In[897]:



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
#     for i,j in zip(B, A):
#         print(i,'\t',j)
    
    result = {}
    for i in A[:k]:
        result[i] = tb[i]
    return result

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
def insert_better_pi_to_deltedTB(TB,dic_pi):
    """pi塞入修改过的tb中"""
    # 遍历dic_pi，将之放入TB中
    
    for index,pi in dic_pi.items():
        if len(TB)<=index:
            TB.append(pi)
        else:
            TB.insert(index,pi)
    return TB


# In[898]:


def crossover(p1,p2,conflict_martix,penalty_par,k):
    """
    input:p1,p2,conflict_martix,k,penalty_rule
    output:new_p1,new_p2
    """
    check_TB_right(p1)
    check_TB_right(p2)
    print('进入交叉...')
     uj = np.random.rand()
    if uj <= Pc:
        # 选出各自最好的k个时间段
        best_kp1 = chose_best_kp_from_tb(p1,penalty_rule,conflict_martix,k)
        best_kp2 = chose_best_kp_from_tb(p2,penalty_rule,conflict_martix,k)

        # 按照对方的k时间段进行旧表考试的删除操作
        p2 = delte_old_course_in_better_pi(p2,best_kp1)
        p1 = delte_old_course_in_better_pi(p1,best_kp2)

        # 将对方的k时间段都插入表，生成交叉之后的表
        new_p1 = insert_better_pi_to_deltedTB(p1,best_kp2)
        new_p2 = insert_better_pi_to_deltedTB(p2,best_kp1)
    else:
        new_p1=p1
        new_p2=p2
    check_TB_right(p1)
    check_TB_right(p2)
    return p1,p2


# In[899]:


# # def crossover(p1,p2):
    
#     check_TB_right(p1)
#     check_TB_right(p2)

#     print('进入交叉...')
#     uj = np.random.rand()
#     if uj <= Pc:
#         #根 据 式 (3-7) 分 别 在 P1  和   P2 中 选 择 min(「ratio*periods1」, 「ratio*periods2」)个连续的时间段长度命名为  Best1, Best2,  转到步骤  2).
#         periods1,periods2 =  len(p1),len(p2)
#         print(periods1,periods2)
#         Best1,Best2 = determinePenForP(p1,int (periods1*ratio)),determinePenForP(p2,int (periods2*ratio))
#         print(Best1,Best2)
#         #交换在 P1  和 P2 中选定时间段中的考试，然后删除重复的考试，在删除过程中，保留下 Best1  和  Best2  中的考试安排。将修复好的 P1,  P2  命名为 PN1, PN2

#         #暂时存储两张旧表,p1,p2
#         temp_p1 = p1.copy()
#         temp_p2 = p2.copy()


#         #在对p1,原表中删去要换的那些课
#         for b in Best2:
#             #遍历best安排的课程
#             for ci in temp_p2[b]:
#                 for pi in p1:
#                     if ci in pi:#在旧表中找到这个课程位置，删去
#                         pi.remove(ci)
#         #                 if pi is None:#如果删到一个都没了，就去掉那个时段
#         #                     p1.remove(pi)
#         for pi in p1:
#             if pi is []:
#                 p1.remove(pi)

#         # #debug:
#         # all_ = np.append(p1,Best2)
#         # all_


#         #         for j in range(periods1):
#         #             if ci in p1[j]:
#         #                 print (len(p1))
#         #                 print (j)

#         #                 p1[j].remove(ci)
#         #                 if p1[j] is not None:
#         #                     #没：删了那个时段
#         #                     #如果删到一个都没了，就去掉那个时段
#         #                     p1.remove(p1[j])
#         # print (p1)

#         #在原表对应位置插入新表 
#         #找到要插入的位置
#         for b in Best2:
#             #遍历best，提取出来并插入原表
#             if b>=len(p1):
#                 np.append(p1,temp_p2[b])
#             else:
#                 np.insert(p1,b,temp_p2[b])



#         #     p1_temp = p1.copy()

#         #     #在p1原表中找到与要换入天中课程冲突的并删去
#         #     for pi in p1:
#         #         for ci in pi:
#         #             for best_day in Best2:
#         #                 if ci in p2[best_day]:#如果原课程在替换课表中找到
#         #                     pi.remove(ci)#删去

#         #     #将要换入的插进对应位置
#         #     for best_day in Best2:
#         #         if best_day>=periods1:
#         #             p1 = np.append(p1,p2[best_day])
#         #         else:
#         #             np.insert(p1,best_day,p2[best_day])


#         #same for p2
#         for b in Best1:
#             #遍历best1安排的课程
#             for ci in temp_p1[b]:
#                 for pi in p2:
#                     if ci in pi:#在旧表中找到这个课程位置，删去
#                         pi.remove(ci)
#         #                 if pi is None:#如果删到一个都没了，就去掉那个时段
#         #                     p1.remove(pi)
#         for pi in p2:
#             if pi is []:
#                 p2.remove(pi)

#         #在原表对应位置插入新表 
#         #找到要插入的位置
#         for b in Best1:
#             #遍历best，提取出来并插入原表
#             if b>=len(p2):
#                 np.append(p2,temp_p1[b])
#             else:
#                 np.insert(p2,b,temp_p1[b])




#     #         for j in range(periods1):
#     #             if ci in p1[j]:
#     #                 print (len(p1))
#     #                 print (j)

#     #                 p1[j].remove(ci)
#     #                 if p1[j] is not None:
#     #                     #没：删了那个时段
#     #                     #如果删到一个都没了，就去掉那个时段
#     #                     p1.remove(p1[j])
#     # print (p1)





#     # #same for p2
#     # #在p2原表中找到与要换入天中课程冲突的并删去
#     # for pi in p2:
#     #     for ci in pi:
#     #         for best_day in Best1:
#     #             if ci in p1_temp[best_day]:#如果原课程在替换课表中找到
#     #                 pi.remove(ci)#删去
#     # #将要换入的插进对应位置
#     # for best_day in Best1: 
#     #     if best_day>=periods2:
#     #         p2 = np.append(p2,p1_temp[best_day])
#     #     else:
#     #         np.insert(p2,best_day,p1_temp[best_day])

#     #         temp_p2 = p2.copy()
#     #         for i in Best1:
#     #             temp_p2[i] = p1[i]
#     #         for j in Best2:
#     #             p1[j] = p2[j]
#     #         p2 = temp_p2
#     #删除重复

#     #     #找到重复，删原来

#     #     #遍历刚换过去的那个时间段
#     #     for i in Best1:
#     #         for ci in p1[i]:
#     #             for j in range(periods1):
#     #                 #遍历原来的时间段
#     #                 if j not in Best1:
#     #                     for cj in p1[j]:
#     #                         if cj in p1[i]:
#     #                             #重复了
#     #                             p1[j].remove(cj)
#     #     for i in Best2:
#     #         for ci in p2[i]:
#     #             for j in range(periods2):
#     #                 if j not in Best2:
#     #                     for cj in p2[j]:
#     #                         if cj in p2[i]:
#     #                             #重复了
#     #                             p2[j].remove(cj)    
#         p1_new,p2_new = p1,p2

#     else:
#         # 否则, PN1=P1, PN2=P2,  转到步骤  3). 
#         p1_new,p2_new = p1,p2
#     # #     nVar=len(p1)
#     # #     gamma= 0
#     # #     for i in range(nVar):
#     # #         uj = np.random.rand()
#     # #         if uj <= 0.5:
#     # #             gamma = (2 * uj) ** (1 / (yita1+1))
#     # #         else:
#     # #             gamma = (1 / (2 * (1 - uj))) ** (1 / (yita1+1))
#     # #         p1[i]=0.5*((1+gamma)*p1[i]+(1-gamma)*p2[i])
#     # #         p2[i]=0.5*((1-gamma)*p1[i]+(1+gamma)*p2[i])
#     # #         p1[i]=min(p1[i],varMax[i])
#     # #         p1[i]=max(p1[i],varMin[i])
#     # #         p2[i]=min(p2[i],varMax[i])
#     # #         p2[i]=max(p2[i],varMin[i])
#     #deubug:检查下生出的解有没有问题
#     check_TB_right(p1_new)
#     check_TB_right(p2_new)
#     print('交叉完毕...')
# #     print ('new_slo_type:',type(p2_new))
#     return p1_new,p2_new
    
    
# #     print('进入交叉...')
    
# #         #根 据 式 (3-7) 分 别 在 P1  和   P2 中 选 择 min(「ratio*periods1」, 「ratio*periods2」)个连续的时间段长度命名为  Best1, Best2,  转到步骤  2).
# #         periods1,periods2 =  len(p1),len(p2)
# #         Best1,Best2 = determinePenForP(p1,periods1*ratio),determinePenForP(p2,periods2*ratio)
        
# #         #交换在 P1  和 P2 中选定时间段中的考试，然后删除重复的考试，在删除过程中，保留下 Best1  和  Best2  中的考试安排。将修复好的 P1,  P2  命名为 PN1, PN2
        
# #         p1_temp = p1.copy()
# #         #在p1原表中找到与要换入天中课程冲突的并删去
# #         for pi in p1:
# #             for ci in pi:
# #                 for best_day in Best2:
# #                     if ci in p2[best_day]:#如果原课程在替换课表中找到
# #                         pi.remove(ci)#删去
       
# #         #将要换入的插进对应位置
# #         for best_day in Best2:
# #             if best_day>=periods1:
# #                 p1 = np.append(p1,p2[best_day])
# #             else:
# #                 np.insert(p1,best_day,p2[best_day])
        
# #         #same for p2
# #         #在p2原表中找到与要换入天中课程冲突的并删去
# #         for pi in p2:
# #             for ci in pi:
# #                 for best_day in Best1:
# #                     if ci in p1_temp[best_day]:#如果原课程在替换课表中找到
# #                         pi.remove(ci)#删去
# #         #将要换入的插进对应位置
# #         for best_day in Best1: 
# #             if best_day>=periods2:
# #                 p2 = np.append(p2,p1_temp[best_day])
# #             else:
# #                 np.insert(p2,best_day,p1_temp[best_day])
        
# # #         temp_p2 = p2.copy()
# # #         for i in Best1:
# # #             temp_p2[i] = p1[i]
# # #         for j in Best2:
# # #             p1[j] = p2[j]
# # #         p2 = temp_p2
# #         #删除重复
        
# #         #找到重复，删原来
        
# #         #遍历刚换过去的那个时间段
# #         for i in Best1:
# #             for ci in p1[i]:
# #                 for j in range(periods1):
# #                     #遍历原来的时间段
# #                     if j not in Best1:
# #                         for cj in p1[j]:
# #                             if cj in p1[i]:
# #                                 #重复了
# #                                 p1[j].remove(cj)
# #         for i in Best2:
# #             for ci in p2[i]:
# #                 for j in range(periods2):
# #                     if j not in Best2:
# #                         for cj in p2[j]:
# #                             if cj in p2[i]:
# #                                 #重复了
# #                                 p2[j].remove(cj)    
# #         p1_new,p2_new = p1,p2
        
# #     else:
# #         # 否则, PN1=P1, PN2=P2,  转到步骤  3). 
# #         p1_new,p2_new = p1,p2
# # #     nVar=len(p1)
# # #     gamma= 0
# # #     for i in range(nVar):
# # #         uj = np.random.rand()
# # #         if uj <= 0.5:
# # #             gamma = (2 * uj) ** (1 / (yita1+1))
# # #         else:
# # #             gamma = (1 / (2 * (1 - uj))) ** (1 / (yita1+1))
# # #         p1[i]=0.5*((1+gamma)*p1[i]+(1-gamma)*p2[i])
# # #         p2[i]=0.5*((1-gamma)*p1[i]+(1+gamma)*p2[i])
# # #         p1[i]=min(p1[i],varMax[i])
# # #         p1[i]=max(p1[i],varMin[i])
# # #         p2[i]=min(p2[i],varMax[i])
# # #         p2[i]=max(p2[i],varMin[i])
# #     #deubug:检查下生出的解有没有问题
# #     check_TB_right(list(p1_new))
# #     check_TB_right(list(p1_new))
    
# #     return list(p1_new),list(p1_new)


# In[900]:


#变异算子-简易版：
def mutate(p):
    index = [i for i in  range(len(p))]
    cindex = np.random.choice(index,2)
    c1,c2 = cindex[0],cindex[1]
    p[c1],p[c2] = p[c2],p[c1]
    return p

#变异算子-原文思路
# def mutate(p,PReinsert,Enum):
    #        '''输入：选择个体 P，变异概率 PReinsert 
#     以及总的考试科目数 Enum. 输出：新产生的个体 PN '''
#        '''输入：选择个体 P，变异概率 PReinsert 
#     以及总的考试科目数 Enum. 输出：新产生的个体 PN '''
#     #     p_len = len(p)
#     #在 P 中随机选择  PReinsert * Enum  门考试，  标记为  Em.  将这些考试在 P 中删除    
#     print ('进入变异...')
#     Em  =[]#用以存储挑出来的课程
#     temp_p = p.copy()#保存副本
#     #随机取特定个要删的课程
#     puping = [i for j in p for i in j]
#     shuffle(puping)
#     citodelte = np.random.choice(puping,int (PReinsert * Enum))
#     print (citodelte)
#     #去原表删下
#     for cs in citodelte:
#         delte_cd_in_p(cs,p)
#         Em.append(cs)
#     #启发式插入
#     # for i in range(int (PReinsert * Enum)):#int (PReinsert * Enum)个课程出来
#     #     pitodelte = np.random.choice(p,1)
#     # print (pitodelte)
#     #     ci2delte = np.random.choice(pitodelte,1)
#     # #         pitodelte.remove(ci2delte)
#     #     p[p.index(list(pitodelte))].remove(ci2delte)
#     #     Em.append(ci2delte)
#     exma_people_dym = exam_people.copy()
#     exma_people_dym = exma_people_dym[Em]
#     exma_people_dym = exma_people_dym.argsort()
#     exma_people_dym = list(exma_people_dym)
#     print (exma_people_dym)

#     while Em:
#         #应用启发式信息将 Em中的考试进行排序。
#         in_c = Em.pop(exma_people_dym.pop())#找到一个待插入的考试序号
#         #看看插在哪儿
#         #伪代码：
#         flag_exist = 0#插了没
#         for pi in p:
#             flag_pi_conflict = 0#pi冲突没
#             for course_arranged in pi:#现有时间表的pi时间段
#                 #pi时间段中的ci考试
#                 #若ci与cj考试冲突，则标记pi发生冲突，并break
#                 if conflict_martix[in_c][course_arranged]:
#                     #冲突
#                     flag_pi_conflict=1
#                     break
#             #判断pi标记，从而得知pi是否能插入
#             if not flag_pi_conflict:
#                 #若可以插入则直接结束此循环
#                 pi.append(in_c)
#                 flag_exist = 1#插了
#                 break
#             #不可以插入则继续看下个pi+1
#         #判断下插进去没，没有的话，就新开一个p放
#         if not flag_exist:
#             p.append([in_c]) 
#     return p
    
#     p_temp = p.deepcopy()
#     shuffle(p_temp)
#     cou_all = []
#     for pi in p_temp:
#         cou_all+=pi
    
#     shuffle(cou_all)
#     cou_all[]
    
#     dj = 0
#     for i in range(len(p)):
#         uj = np.random.rand()
#         if uj < 0.5:
#             dj = (2 * uj) ** (1 / (yita2+1)) - 1
#         else:
#             dj = 1 - (2 * (1 - uj)) ** (1 /(yita2+1))
#         p[i] = p[i] + dj
#         p[i]=min(p[i],varMax[i])
#         p[i]=max(p[i],varMin[i])
    
#     return pn


# In[901]:


def cross_mutation(p1,p2):
    #交叉变异,不拷贝的话原始数据也变了
    print ('进入交叉变异...')
    y1=np.copy(p1)
    y2=np.copy(p2)
    y1,y2=crossover(y1,y2)
    Debug_bianyi_P.append(y1)
    Debug_bianyi_P.append(y2)
    if np.random.rand()<PM:
        mutate(y1)
    if np.random.rand()<PM:
        mutate(y2)
    return y1,y2



#交叉生成下个解
def generate_next(idx,xk,xl,fitness):
    print ('产生下一个解...')
    y0,y1=cross_mutation(xk,xl)
#     y0=cross_mutation2(xk,xl)
    #对y进行修复根据约束
#     for i in range(nVar):
#         y0[i]=max(varMin[i],y0[i])
#         y0[i]=min(varMax[i],y0[i])
# #     return y0
#         y1[i]=max(varMin[i],y1[i])
#         y1[i]=min(varMax[i],y1[i])
    fx1=np.array(fitness(y0))
    fx2=np.array(fitness(y1))
    if isDominates(fx1,fx2):
        return y0
    elif isDominates(fx2,fx1):
        return y1
    else:
        if np.random.rand()<0.5:
            return y0
        else:
            return y1


# In[902]:


#更新邻居        
def update_neighbor(idx,y):
    print ('update_neighbor...')
    #若gy<gx更新,用的权重是邻居的权重
    Bi=sp_neighbors[idx]
    fy=y.cost
    for j in range(len(Bi)):
        w=Lambda[Bi[j]]
        maxn_y=max(w*abs(fy-z))
        maxn_x=max(w*abs(pops[Bi[j]].cost-z))
        if maxn_x>=maxn_y:
            pops[Bi[j]]=y


# In[903]:


#创建冲突矩阵
crs_n = len(crsdata)
stu_n = len(studata)
conflict_martix = np.zeros((crs_n,crs_n),dtype = int)


def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        l = list(c)
        temp_list2.append(l)
    return temp_list2

for l1 in studata:
    for l2 in combine(l1,2):
        i , j = l2[0] ,l2[1]
        conflict_martix[i-1][j-1] += 1
        conflict_martix[j-1][i-1] += 1

#考试科目冲突矩阵，cij表示需要共同参加i,j考试的学生人数
conflict_martix


#每门课的学生人数列表
exam_people = np.zeros((crs_n,),dtype=int)
for l1 in studata:
    for exnum in l1:
        exam_people[exnum-1] += 1
exam_people
    


# In[904]:


#适应性函数
def object_function(TB):
    #DB检查
    check_TB_right(TB)
    
    P = len(TB)
    pen = 0#罚值初值
    #将每个学生的data转为课程组合列表
    e1_p = 0
    e2_p = 10
    for l1 in studata:
        for l2 in combine(l1,2):
            #对于每个课程组合，把其在tb上的距离找到
            e1,e2 = l2[0],l2[1] 
            for pi in TB:
#                 print(TB,pi)
                #TODO：这有个bug,应该是由解的不可行引起。待调
                try:
                    posi = list(TB).index(pi)
                    if e1 in pi:
                        e1_p = posi
                    if e2 in pi:
                        e2_p = posi
                except:
                    e2_p=5
                    e1_p=10
            delta_12 = abs(e2_p-e1_p)
            #并算出其罚值，累加起来再规整
            pen += calculate_penalty(delta_12)
    pen/=stu_num#归一化处理
    return [P,pen]#适应函数


# In[905]:


#定义种群类
class pop():
    def __init__(self,var,cost):
        self.var=var
        self.cost=cost
        self.dominate=False


# In[906]:


#初始化种群
def initPop(npop,mode,fitness):
    global pops
    pops=[]
    for i in range(npop):
        print('正在生成解{}'.format(i))
#         if i<=npop/2:
#         if i<=npop:
            #随机安排考试顺序，生成一半的解
    #初始变量
    #空时间安排表var
    #考试列表crs，并做下打乱
    #把crs按一定规则给塞入var中
        temp_var = []
        crs = [x for x in range(0, crs_n)]
        random.shuffle(crs)
        temp_var.append([crs.pop()])
        while crs:
            #循环，直到把所有考试都安排好
            in_c = crs.pop()#找到一个待插入的考试序号
            #看看插在哪儿
            #伪代码：
            flag_exist = 0#插了没
            for pi in temp_var:
                flag_pi_conflict = 0#pi冲突没
                for course_arranged in pi:#现有时间表的pi时间段
                    #pi时间段中的ci考试
                    #若ci与cj考试冲突，则标记pi发生冲突，并break
                    if conflict_martix[in_c][course_arranged]:
                        #冲突
                        flag_pi_conflict=1
                        break
                #判断pi标记，从而得知pi是否能插入
                if not flag_pi_conflict:
                    #若可以插入则直接结束此循环
                    pi.append(in_c)
                    flag_exist=1#插了
                    break
                #不可以插入则继续看下个pi+1
            #判断下插进去没，没有的话，就新开一个p放
            if not flag_exist:
                temp_var.append([in_c]) 
        var=temp_var
        #print (var)
        cost=fitness(var)
        pops.append(pop(var,cost))

    
        
#     if i>npop/2:
#         #启发安排考试顺序
#         temp_var = []
#         exam_people2arrange=exam_people.copy()
#         crs = [x for x in range(0, crs_n)]
#         #规则一：Largest Degree (LD):  先排那些与其它考试冲突数最多的考试
#         #规则三：Saturation Degree (SD):  先排那些在现有考试时间表中在不触犯硬约束的前提下能够排进去的时间段数最少的考试。
        
        
#         #规则二：Largest Weighted Degree (LWD):  与 LD 类似，但是先排那些涉及的学生人数比较多的，而不是冲突数最多的考试科目
#         in_c = LWD(exam_people2arrange)
#         del exam_people2arrange[in_c]
#         temp_var.append([in_c])
#         k=0
        
#         #按规则二生成一个安排表解
#         while exam_people2arrange:
#             #循环，直到把所有考试都安排好
#             in_c = LWD(exam_people2arrange)#找到相关人数最多的考试序号
#             del exam_people2arrange[in_c]#删去
#             #看看插在哪儿
            
#             #伪代码：
#             flag_exist = 0#插了没
#             for pi in temp_var:
#                 flag_pi_conflict = 0#pi冲突没
#                 for course_arranged in pi:#现有时间表的pi时间段
#                     #pi时间段中的ci考试
#                     #若ci与cj考试冲突，则标记pi发生冲突，并break
#                     if conflict_martix[in_c][course_arranged]:
#                         #冲突
#                         flag_pi_conflict=1
#                         break
#                 #判断pi标记，从而得知pi是否能插入
#                 if not flag_pi_conflict:
#                     #若可以插入则直接结束此循环
#                     pi.append(in_c)
#                     flag_exist=1#插了
#                     break
#                 #不可以插入则继续看下个pi+1
#             #判断下插进去没，没有的话，就新开一个p放
#             if not flag_exist:
#                 temp_var.append([in_c])
                
                
#     print (temp_var)

# def LD(has_arrange):   
#     '''先排那些与其它考试冲突数最多的考试'''
#     max_c_cor = 0
#     max_connum = sum(conflict_martix[0])
#     for i range(crs_n-1):
#         if i in has_arrange:
#             continue
#         elif sum(conflict_martix[i])>max_connum:
#             max_connum = sum(conflict_martix[i])
#             max_c_cor = i
#     return max_c_cor

# def LD(course2arrange):   
#     '''先排那些与其它考试冲突数最多的考试'''
#     max_c_cor = 0
#     max_connum = sum(conflict_martix[0]!=0)
#     for i in course2arrange:
#         if sum(conflict_martix[i]!=0)>max_connum:
#             max_connum = sum(conflict_martix[i]!=0)
#             max_c_cor = i
#     return max_c_cor

# def SD(temp_P,course2arrange):
#     '''先排那些在现有考试时间表中在不触犯硬约束的前提下能够排进去的时间段数最少的考试'''
#     course_insert_pi_number = {}
#     for course_code in  course2arrage:
#         #遍历pi
#         num_p = 0
#         for pi in temp_P:
#             for other_course in pi:
#                 if conflict_martix[course_code][other_course]:
#                     break
#                 num_p+=1
#         course_insert_pi_number[course_code]=num_p
#     key_min = min(course_insert_pi_number.keys(), key=(lambda k: course_insert_pi_number[k]))  
#     return int(key_min)

# def LWD(exam_people2arrange):
#     '''与 LD 类似，但是先排那些涉及的学生人数比较多的，而不是冲突数最多的考试科目'''
#     return exam_people2arrange.index(exam_people2arrange.max())



# In[907]:


#生成均分的权重向量
def genVector2(nObj,npop,T):
    Lambda=[]
    dist=np.zeros((npop,npop))
    for i in range(npop):
        w=np.random.rand(nObj)
        w=w/np.linalg.norm(w)
#         w=w/sum(w)
        Lambda.append(w)
    for i in range(npop-1):
        for j in range(i+1,npop):
            dist[i][j]=dist[j][i]=np.linalg.norm(Lambda[i] - Lambda[j])
    #TODO:将P时间段信息也嵌入
    sp_neighbors=np.argsort(dist,axis=1)
    sp_neighbors=sp_neighbors[:,:T]
    return Lambda, sp_neighbors


# sp_neighbors


# In[908]:


#均分的权重向量，令居列表初始化，种群初始化，设置参考点，创建初代精英种群EPset

Lambda,sp_neighbors=genVector2(nObj,population,T)#均分的权重向量和邻居列表

initPop(population,'random',object_function)#种群初始化
print('种群初始化完毕')
z=[0,0]
for p in range(population):
    for j in range(nObj):
        z[j]=min(pops[p].cost[j],z[j])
z=np.array(z)

print('理想解设置完毕,目前为{}'.format(z))


determinDomination(pops)
EPs=copy.deepcopy([x for x in pops if x.dominate!=True])

print('EPs设置完毕,目前为{}'.format(EPs))

#DEBUG：检查生成的解符不符合要求
for i in range(len(pops)):
    check_TB_right(pops[i].var)


# In[ ]:


#循环：
    #交叉变异
    #更新领域，z和EP
if __name__ == "__main__":
    start = time.time()
    for j in range(iteration):
        if j % 10 == 0:
            print("=" * 10, j, "=" * 10)
            print('当前理想解为',z, ',当前EPS大小为',len(EPs))
        for i in range(population):
            #选出个体的邻居
            Bi = sp_neighbors[i]
            choice = np.random.choice(T, 2, replace=False)  # 选出来的邻居应该不重复
#             print (k,l,len(pops))
            k = Bi[choice[0]]
            l = Bi[choice[1]]
            xk = pops[k]
            xl = pops[l]
            # 产生新的解，并对解进行修复
            print('在第',i,'个体附近选择了',k,'和',l,'个体作为交配个体')
            y = generate_next(i, xk.var, xl.var, object_function)
            fv_y = np.array(object_function(y))
            y = pop(y, fv_y)
            print('产生新解完毕',y.var,fv_y)
            
            
            # 更新z,
            t = z > fv_y
            z[t] = fv_y[t]
            # 更新邻域解
            update_neighbor(i, y)
            ep = False
            delete = []
            for k in range(len(EPs)):
                if (fv_y == EPs[k].cost).all():  # 如果有一样的就不用算了啊
                    ep = True
                    break
                if isDominates(fv_y, EPs[k].cost):
                    #TODO:删不够咋办？
                    delete.append(EPs[k])
                elif ep == False and isDominates(EPs[k].cost, fv_y):
                    ep = True
                    break  # 后面就不用看了，最好也是互不支配
            if len(delete) != 0:
                for k in range(len(delete)):
                    EPs.remove(delete[k])
            if ep == False:
                EPs.append(y)
            while len(EPs) > archive:
                #TODO:优化？
                select = np.random.randint(0, len(EPs))
                del EPs[select]
    #         if len(EPs)>archive:
    #             delete=np.random.choice(EPs,len(EPs)-archive)
    #             for k in range(len(delete)):
    #                 EPs.remove(delete[k])
    end = time.time()
    print("循环时间：%2f秒" % (end - start))

    


# In[ ]:


#结果可视化：
# set(list(range(1,crs_n)))
plt.title("scatter diagram.") 
plt.xlim(xmax=100,xmin=0)
plt.ylim(ymax=100,ymin=0)
plt.xlabel("f1")
plt.ylabel("f2")
for eps in EPs:
    plt.plot(eps.cost,'ro')
plt

