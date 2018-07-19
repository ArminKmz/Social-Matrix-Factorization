def read(filename, path='dataset/'):
    import numpy as np
    return np.loadtxt(path+filename,  delimiter=',')

def center(data):
    '''
        center data and return mean
    '''
    import numpy as np
    mean = np.mean(data[:, 2])
    data[:, 2] -= mean
    return mean

def data_to_list(N, M, data):
    '''
        data -> user, item, score
        return list of items rated by user &
               list of users rated item.
    '''
    user_list = [[] for _ in range(N+1)]
    item_list = [[] for _ in range(M+1)]
    for i in range(data.shape[0]):
        u_id = int(data[i][0])
        i_id = int(data[i][1])
        score = data[i][2]
        user_list[u_id].append([i_id, score])
        item_list[i_id].append([u_id, score])
    return user_list, item_list

def plot(x, y, xlabel, ylabel, title, color):
    import matplotlib.pyplot as plt
    plt.plot(x, y, color=color)
    plt.plot(x, y, color+'o')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def edge_list_to_adj_list(MAX_ID, edge_list, LIMIT=-1):
    '''
        MAX_ID -> maximum id of users
        LIMIT -> maximum outdegree
                 for default value there is no limit
        return G, G^T
    '''
    import numpy as np
    adj = [[] for _ in range(MAX_ID+1)]
    adjT = [[] for _ in range(MAX_ID+1)]
    for l in edge_list:
        [u, v] = l
        u, v = int(u), int(v)
        if max(u, v) <= MAX_ID and (LIMIT == -1 or len(adj[u]) < LIMIT):
            adj[u].append(v)
            adjT[v].append(u)
    return adj, adjT
