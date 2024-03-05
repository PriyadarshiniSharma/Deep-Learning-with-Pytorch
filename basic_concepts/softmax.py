import torch
import torch.nn as nn
import numpy as np


loss = nn.CrossEntropyLoss()

y = torch.tensor([2, 0, 1])
# nsamples * nclasses = 3 x3
yp_good = torch.tensor([[.1, 1.0, 2.1], [2.0, 1.0, 0.1], [.1, 3.0, 0.1]])
yp_bad = torch.tensor([[2.1, 1.0, 0.1], [.1, 1.0, 2.1], [.1, 3.0, 0.1]])

l1 = loss(yp_good, y)
l2 = loss(yp_bad, y)
print(l1)
print(l2)

_, prediction1 = torch.max(yp_good, 1)
_, prediction2 = torch.max(yp_bad, 1)

print(prediction1)
print(prediction2)



# def cross_entropy(actual, predicted):
#     loss = -np.sum(actual * (np.log(predicted)))
#     return loss

# y = np.array([1, 0, 0])
# yp_good = np.array([.7, .2, .1])
# yp_bad = np.array([.1, .3, .6])

# l1 = cross_entropy(y, yp_bad)
# l2 = cross_entropy(y, yp_good)
# print(l1)
# print(l2)


# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis = 0)

# x = np.array([2.0, 1.0, 0.1])
# outputs = softmax(x)
# print(outputs)

# x = torch.tensor([2.0, 1.0, 0.1])
# outputs = torch.softmax(x, dim = 0)
# print(outputs)