import utils, SMF
import numpy as np

train_data = utils.read('train_user_item_score.txt')
valid_data = utils.read('validation_user_item_score.txt')

edge_list = utils.read('users_connections.txt')

MAX_ID = int(np.max(train_data, axis=0)[0])
LIMIT = 1

network, network_transpose = utils.edge_list_to_adj_list(MAX_ID, edge_list, LIMIT)

train_mean = utils.center(train_data)
valid_data[:, 2] -= train_mean

# ----------------------------------
#         train Model
# ----------------------------------
sigma = 1
sigma_u = 1
sigma_v = 1
sigma_w = 1
D = 1
iterations = 10

model = SMF.Model(train_data, network, network_transpose, D, sigma, sigma_u,
                    sigma_v, sigma_w, iterations)
rmse_list = model.train()

x = [i for i in range(1, iterations+1) if i % 2 == 0 or (i == 1)]
y = [rmse_list[i] for i in range(1, iterations+1) if i % 2 == 0 or (i == 1)]
print("RMSE on validation data:", model.test(valid_data))
utils.plot(x, y, 'iterataions', 'RMSE', 'RMSE per iterations', 'b')

# ----------------------------------
#           load Model
# ----------------------------------
# set test data

# test_data =
# model = SMF.Model(network=network, load=True)
# print("RMSE on test data:", model.test(test_data))
