import geomloss
import torch

loss='sinkhorn'
cost_function='euclidean'
p=2
debias=False
entreg=1e-1
device='cuda:3'

X1 = torch.load("/home/netlab/DL_lab/otdd/X1.pt")
X2 = torch.load("/home/netlab/DL_lab/otdd/X2.pt")
Y1 = torch.load("/home/netlab/DL_lab/otdd/Y1.pt")
Y2 = torch.load("/home/netlab/DL_lab/otdd/Y2.pt")
c1 = torch.load("/home/netlab/DL_lab/otdd/c1.pt")
c2 = torch.load("/home/netlab/DL_lab/otdd/c2.pt")
n1, n2 = len(c1), len(c2)

small_cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
big_cost_function = "(SqDist(X,Y) / IntCst(2))"

small_distance = geomloss.SamplesLoss(
    loss=loss, p=p,
    cost=small_cost_function,
    debias=debias,
    blur=entreg**(1 / p),
)

big_distance = geomloss.SamplesLoss(
    loss=loss, p=p,
    cost=big_cost_function,
    debias=debias,
    blur=entreg**(1 / p),
)

i = 5 # 5
j = 6
temp_left = X1[Y1==c1[i]].to(device)
temp_right = X2[Y2==c2[j]].to(device)
if temp_left.shape[0] * temp_right.shape[0] >= 5000 ** 2:
    print(big_distance(temp_left, temp_right).item())
else:
    print(small_distance(temp_left, temp_right).item())