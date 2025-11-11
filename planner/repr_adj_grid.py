import numpy as np
from copy import deepcopy
from queue import PriorityQueue
import tqdm
import joblib
import sys
sys.path.append('..')
from algos.trajectory_buffer import TrajectoryReprBuffer


class NodeDistPair:
    def __init__(self, idx, weight):
        self.idx = idx
        self.weight = weight
    
    def __lt__(self, p):
        return self.weight < p.weight


'''
Data Structure: ReprAdjGrid
components: adjacency matrix, hashmap, node index list
representations elements may be negative, so a map is needed for construction of adjacency matrix
we search for shortest path in hash index space and then use the path weights in repr space for subgoal choosing
'''
class ReprAdjGrid:
    
    def __init__(self, scale=1., data=None, max_nodes=3500):
        self.scale = scale
        self.repr_dict = dict()  # repr -> nodeID
        self.node_dict = dict()  # nodeID -> repr
        self._adj_mat = np.inf * np.ones((max_nodes, max_nodes), dtype=np.int32)
        self._dist_mat = np.inf * np.ones((max_nodes, max_nodes), dtype=np.float32)
        self._shortest_dist = np.inf * np.ones((max_nodes, max_nodes), dtype=np.float32)
        for i in range(max_nodes):
            self._dist_mat[i, i] = 0
            self._adj_mat[i, i] = 1
        self._max_nodes = max_nodes
        
        if data:
            self.insert_repr_trajectories(data)
            self.update_distance_matrix_prioritized_parallel()
        
    '''
    trajs: samples from trajectory buffer, trajs: List[List] of state / subgoal representations
    '''
    def insert_repr_trajectories(self, trajs):
        if not trajs:
            return 
        print('processing traj data...')
        self._num_nodes = 0
        for traj in trajs:
            for i in range(len(traj) - 1):
                self.add_edge(repr1=traj[i], repr2=traj[i+1], weight=1.)
                if self._num_nodes >= self._max_nodes:
                    return
            '''
            for i in range(len(traj)):
                for j in range(i+1, len(traj)):
                    self.add_edge(repr1=traj[i], repr2=traj[j], weight=np.abs(np.float32(j - i)))
                    if self._num_nodes >= self._max_nodes:
                        return
            '''
            
    # path weight of a specified edge, default 1, can be modified by other metrics e.g. temporal distance, etc
    def _path_weight(self):
        return 1
    
    def add_edge(self, repr1, repr2, weight):
        if weight > 1:
            return
        xy1, idx1 = self.query_node(repr1)
        xy2, idx2 = self.query_node(repr2)
        if idx1 == -1:
            self.repr_dict[xy1] = self._num_nodes
            self.node_dict[self._num_nodes] = (xy1[0] * self.scale, xy1[1] * self.scale)
            self._num_nodes += 1
        if idx2 == -1:
            self.repr_dict[xy2] = self._num_nodes
            self.node_dict[self._num_nodes] = (xy2[0] * self.scale, xy2[1] * self.scale)
            self._num_nodes += 1
        # use minimal temporal interval for direct distance here
        tmp = self._dist_mat[self.repr_dict[xy1], self.repr_dict[xy2]]
        self._dist_mat[self.repr_dict[xy1], self.repr_dict[xy2]] = min(weight, tmp)
        tmp = self._dist_mat[self.repr_dict[xy2], self.repr_dict[xy1]]
        self._dist_mat[self.repr_dict[xy2], self.repr_dict[xy1]] = min(weight, tmp)
    
    # return node index corresponding to repr
    def query_node(self, repr):
        hash_x, hash_y = int(repr[0] // self.scale), int(repr[1] // self.scale)
        return (hash_x, hash_y), self.repr_dict.get((hash_x, hash_y), -1)
    
    # return distance between two points in repr space
    def query_dist(self, repr_start, repr_end):
        _, start_idx = self.query_node(repr_start)
        _, end_idx = self.query_node(repr_end)
        if start_idx == -1 or end_idx == -1:
            return np.inf
        return self._shortest_dist[start_idx, end_idx]
    
    # 从距离大于0.9max的节点中任选一个
    def query_max_dist_from(self, repr_start):
        _, start_idx = self.query_node(repr_start)
        if start_idx != -1:
            tmp = deepcopy(self._shortest_dist[start_idx, :])
            for i in range(tmp.shape[0]):
                if tmp[i] == np.inf:
                    tmp[i] = -1
            # idx = np.argmax(tmp)
            idx = np.random.choice(np.where(tmp > (np.max(tmp) * 0.9))[0])
            return self.node_dict[idx]
        return ()
    
    # bfs不一定能确保最短，考虑使用带trace的dijkstra实现，迭代结束后反查路径
    def __route_helper_dijkstra(self, start_idx: int, end_idx: int) -> np.ndarray:
        if self._shortest_dist[start_idx, end_idx] == np.inf:
            return np.empty((0, 2))
        _, trace = self._search_dijkstra(start_idx)
        route = []
        cur = end_idx
        while cur != start_idx:
            route.append(list(self.node_dict.get(cur)))
            cur = trace[cur]
        route.append(list(self.node_dict.get(start_idx)))
        route.reverse()
        return np.array(route)

    def query_route_parallel(self, starts, ends):
        res = joblib.Parallel(n_jobs=joblib.cpu_count()-5)(joblib.delayed(self.query_route)(pt[0], pt[1]) for pt in zip(starts, ends))
        return res
    
    def query_route(self, start_repr, end_repr):
        _, start_idx = self.query_node(start_repr)
        _, end_idx = self.query_node(end_repr)
        if start_idx == -1 or end_idx == -1:
            return np.empty((0, 2))
        if self.query_dist(start_repr, end_repr) == np.inf:
            return np.empty((0, 2))
        return self.__route_helper_dijkstra(start_idx, end_idx)
    
    def query_dist_multi(self, repr_start, reprs_end: np.ndarray):
        ans = np.ones(reprs_end.shape[0]) * np.inf
        _, start_idx = self.query_node(repr_start)
        if start_idx != -1:
            # simple loop (may be slow)
            for i in range(reprs_end.shape[0]):
                _, end_idx = self.query_node(reprs_end[i, :])
                if end_idx != -1:
                    ans[i] = self._shortest_dist[start_idx, end_idx]
        return ans
    
    # cur, start, end are all reprs
    def select_waypoint(self, cur, start, end):
        if not start or not end:
            raise ValueError("illegal start or end point")
        route = self.query_route(start, end)  # (route_length, 2)
        dist_to_route = np.linalg.norm(route - cur, axis=-1)
        nearest_idx = np.argmin(dist_to_route)
        print("below results are from waypoint selection")
        print(nearest_idx)
        print(route[nearest_idx])
    
    # get average distance from 'repr' to all other recorded nodes
    # calculate on short path graph, omitting inf distances
    def query_avg_dist_obsoleted(self, repr):
        _, idx = self.query_node(repr)
        return np.mean(self._dist_mat[idx, np.where(self._dist_mat[idx, :] != np.inf)[0]])
    
    def query_avg_dist(self, repr):
        _, idx = self.query_node(repr)
        if idx != -1:
            return np.mean(self._shortest_dist[idx, np.where(self._shortest_dist[idx, :] != np.inf)[0]])
        return np.inf
    
    def query_avg_dist_multi(self, reprs):
        return np.array([self.query_avg_dist(reprs[i, :]) for i in range(reprs.shape[0])])
    
    # get proportion of nodes that have inf distance with queried node 'repr'
    def query_inf_prop_obsoleted(self, repr):
        _, idx = self.query_node(repr)
        return len(np.where(self._dist_mat[idx, :] == np.inf)[0]) / len(self._dist_mat[idx, :])

    def query_inf_prop(self, repr):
        _, idx = self.query_node(repr)
        if idx != -1:
            return len(np.where(self._shortest_dist[idx, :] == np.inf)[0]) / len(self._shortest_dist[idx, :])
        return 1.
    
    def query_inf_prop_multi(self, reprs):
        return np.array([self.query_inf_prop(reprs[i, :]) for i in range(reprs.shape[0])])
    
    # run standard Floyd algorithm for shortest path searching
    def update_distance_matrix(self):
        print('updating distances for {} nodes...'.format(self._num_nodes))
        for k in range(self._num_nodes):
            for i in range(self._num_nodes):
                for j in range(self._num_nodes):
                    self._dist_mat[i, j] = min(self._dist_mat[i, j], self._dist_mat[i, k] + self._dist_mat[k, j])
        self._shortest_dist = deepcopy(self._dist_mat)
        
    # accelerate shortest path searching by replacing Floyd algorithm with prioritized Dijkstra algorithm
    def update_distance_matrix_prioritized(self):
        print('updating distances for {} nodes'.format(self._num_nodes))
        for start_node in tqdm.tqdm(range(self._num_nodes)):
            dist_list, _ = self._search_dijkstra(start_node)
            self._shortest_dist[start_node, :self._num_nodes] = deepcopy(dist_list)   
        # print(self._shortest_dist[:self._num_nodes, :self._num_nodes])         

    def update_distance_matrix_prioritized_parallel(self):
        # dist_lists: list[np.ndarray]
        print('UPDATING distances for {} nodes'.format(self._num_nodes))
        res_lists = joblib.Parallel(n_jobs=(joblib.cpu_count()-5))(joblib.delayed(self._search_dijkstra)(idx) for idx in range(self._num_nodes))
        dist_lists = [res[0] for res in res_lists]
        self._shortest_dist[:self._num_nodes, :self._num_nodes] = deepcopy(np.array(dist_lists))
    
    # single-source shortest path searching by prioritized Dijkstra algorithm
    def _search_dijkstra(self, start_idx):
        dist_list = np.inf * np.ones(self._num_nodes)
        dist_list[start_idx] = 0.
        trace = [-1 for _ in range(self._num_nodes)]  # trace[i]: i节点的前置节点
        trace[start_idx] = start_idx
        queue = PriorityQueue()
        visit = [False for _ in range(self._num_nodes)]
        queue.put(NodeDistPair(start_idx, 0))
        
        while not queue.empty():
            min_dist_idx = queue.get().idx
            # print(min_dist_idx)
            if visit[min_dist_idx]:
                continue
            visit[min_dist_idx] = True

            for i in range(self._num_nodes):
                if visit[i] == False and dist_list[min_dist_idx] + self._dist_mat[min_dist_idx, i] < dist_list[i]: 
                    dist_list[i] = dist_list[min_dist_idx] + self._dist_mat[min_dist_idx, i]
                    trace[i] = min_dist_idx
                    queue.put(NodeDistPair(i, dist_list[i]))
        
        return deepcopy(dist_list), deepcopy(trace)
    
    # export all nodes
    def export_node_array(self) -> np.ndarray:
        _exported_res = list(self.node_dict.values())
        return np.array([[tp[0], tp[1]] for tp in _exported_res])
    
def _adj_grid_test():
    
    max_coord = 30
    
    def gen_random_movement():
        x, y = np.random.randint(-1, 2), np.random.randint(-1, 2)
        while x == 0 and y == 0:
            x, y = np.random.randint(-1, 2), np.random.randint(-1, 2)
        return x, y
    
    buffer = TrajectoryReprBuffer(capacity=200)
    for _ in range(100):
        buffer.create_new_trajectory()
        coord = [np.random.randint(max_coord + 1), np.random.randint(max_coord + 1)]
        buffer.append(coord)
        for step in range(200):
            # generate random walk trajs
            move_x, move_y = gen_random_movement()
            new_x, new_y = coord[0] + move_x, coord[1] + move_y
            while new_x < 0 or new_x > max_coord or new_y < 0 or new_y > max_coord:
                move_x, move_y = gen_random_movement()
                new_x, new_y = coord[0] + move_x, coord[1] + move_y
            coord = [new_x, new_y]
            buffer.append(coord)
    
    grid = ReprAdjGrid(scale=1., data=buffer.trajectory)
    print(grid.query_dist([0, 0], [max_coord, max_coord]))
    print(grid.query_route(start_repr=[0, 0], end_repr=[max_coord, max_coord]))
    print(grid._num_nodes)
    grid.select_waypoint(cur=[20, 17], start=[0, 0], end=[max_coord, max_coord])

def _scale_test():
    max_coord = 30
    
    def gen_random_movement():
        x, y = np.random.randint(-1, 2), np.random.randint(-1, 2)
        while x == 0 and y == 0:
            x, y = np.random.randint(-1, 2), np.random.randint(-1, 2)
        return x, y
    
    buffer = TrajectoryReprBuffer(capacity=200)
    for _ in range(100):
        buffer.create_new_trajectory()
        coord = [np.random.randint(max_coord + 1), np.random.randint(max_coord + 1)]
        buffer.append(coord)
        for step in range(200):
            # generate random walk trajs
            move_x, move_y = gen_random_movement()
            new_x, new_y = coord[0] + move_x, coord[1] + move_y
            while new_x < 0 or new_x > max_coord or new_y < 0 or new_y > max_coord:
                move_x, move_y = gen_random_movement()
                new_x, new_y = coord[0] + move_x, coord[1] + move_y
            coord = [new_x, new_y]
            buffer.append(coord)
    
    grid = ReprAdjGrid(scale=2, data=buffer.trajectory)
    print(grid.query_dist([0, 0], [max_coord, max_coord]))
    print(grid.query_route(start_repr=[0, 0], end_repr=[max_coord, max_coord]))
    print(grid._num_nodes)
    grid.select_waypoint(cur=[20, 17], start=[0, 0], end=[max_coord, max_coord])
    print(grid.query_max_dist_from([0, 0]))
    
if __name__ == '__main__':
    _scale_test()
    