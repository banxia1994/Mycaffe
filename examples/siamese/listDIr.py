#conding:utf-8
import os
'''
dir = '/data/Fingervein/MMCBNU-6000/ROI_rectangle/'

dirs = os.listdir(dir)

with open('../train.txt','w') as train,open('../test.txt','w') as test:
    for i in dirs:
        ids = int(i.split('.')[0])
        if ids%10 == 1:
            test.writelines(i+' '+ str((ids-1)//10)+'\n')
        else:
            train.writelines(i+' '+ str((ids-1)//10)+'\n')

'''
import random
def genrandPair(name,num,line_num):
    with open('../{}.txt'.format(name),'w') as pair:
        count = 0
        for  i in range(line_num):
            pair.writelines(str(random.randint(0,num-1))+' '+str(random.randint(0,num-1))+'\n')
            if ((i+1)%3==0):
                pair.writelines(str(count)+' '+str(count)+'\n')
                count += 1


#genrandPair('testpair',600,600*3)
genrandPair('trainpair',5400,5400*3)
#print (random.randint(0,600-1))
