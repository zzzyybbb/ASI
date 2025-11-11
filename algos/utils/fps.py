import torch
import numpy as np
from copy import deepcopy

# merge new representations into origin representations
def _merge_unique_arr(origin_arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
    assert len(new_arr.shape) == 2
    origin_list = [tuple(pt) for pt in origin_arr] if len(origin_arr) > 0 else []
    new_list = [tuple(pt) for pt in new_arr]
    unique_res = list(set(origin_list + new_list))
    return np.array([[pt[0], pt[1]] for pt in unique_res])

# fps
def farthest_point_sample(data, K=1000, basis=None, eps=1e-3, inf=100000, device="cpu", verbose=False):
    input_np = 0
    org_data = deepcopy(data)
    if isinstance(data, np.ndarray):
        data = torch.Tensor(data).to(device)
        input_np = 1

    data_ = data.view(len(data), -1)
    dist = torch.zeros(len(data),).to(data.device) + inf

    if basis is not None:
        basis = basis.view(len(basis), -1)
        new_dist = ((data_[:, None, :] - basis[None, :])
                    ** 2).mean(dim=2).min(dim=1)[0]
        dist = torch.stack((dist, new_dist)).min(dim=0)[0]

    choosed = []
    while len(choosed) < K:
        if dist.max() < eps:
            break
        idx = dist.argmax()
        new = data[idx]
        choosed.append(idx)
        new_dist = ((data_ - new.view(-1)[None, :])**2).mean(dim=1)
        dist = torch.stack((dist, new_dist)).min(dim=0)[0]
    if len(choosed) == 0:
        return []
    if verbose:
        print('Found {} points'.format(len(choosed)))
    choosed = torch.stack(choosed)
    if input_np:
        choosed = choosed.detach().cpu()
    return org_data[choosed, :]

if __name__ == '__main__':
    data = np.random.randn(1000, 2)
    choosed = farthest_point_sample(data=data, K=60)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c='green', s=0.4)
    plt.scatter(choosed[:, 0], choosed[:, 1], c='red', s=3)
    plt.savefig('test.png')
    plt.close()
    
    print("test merge")
    org_arr = np.array([[1, 2], [3, 4], [1, 2], [4, 5], [3, 4]])
    new_arr = np.array([[4, 5], [3, 4], [1, 3], [2, 6]])
    print(_merge_unique_arr(org_arr, new_arr))
    
    