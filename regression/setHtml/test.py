# coding:utf-8
import math
def sum(inputint):
    judge=False
    all_sum=0
    range_of_time=int(math.sqrt(inputint))

    for i in range(1,range_of_time):
        all_sum=i*i
        all_sum+=i

        if all_sum%2==1:
            continue

        if (all_sum/2)>inputint:
            continue

        if (inputint-(all_sum/2))==0:
            judge=True
            break
    return judge