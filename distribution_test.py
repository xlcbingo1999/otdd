import math
c = 0.02
k = 0.1
log_func = lambda p, p_max: math.log(p) / math.log(p_max)
log_func_2 = lambda p, p_max: 1 - math.exp(-1 / (p * c)) / math.exp(-1 / (p_max * c))
sigmold_func = lambda p, p_max: 1 / (1 + math.exp(-p))
# g_func = lambda p, p_max: 1 / (1 + math.exp( (log_func_2(p, p_max) - 0.5)))

ps = [40, 50, 80, 120, 200, 500, 1000, 1200, 1500, 2000]

for index, p in enumerate(ps):
    p_max = max(ps[0:index+1])
    print("ps[0:index]: {}".format(p_max))
    for temp_index in range(index+1):
        print("p: {} => g_func: {}".format(ps[temp_index], log_func_2(ps[temp_index], p_max)))
    print("\n\n")