import numpy as np
from numpy.linalg import inv
import utils
import pickle
import os

class Model:
    def __init__(self, train_data=None, network=None, network_T=None, D=1, sigma=1,
                sigma_u=1, sigma_v=1, sigma_w=1, iterations=10, load=False):
        if load:
            self.mean_U = np.load('model/E_U.npy')
            self.mean_V = np.load('model/E_V.npy')
            self.D = self.mean_U.shape[1]
            self.network = network
            with open('model/E_W.txt', 'rb') as fp:
                self.mean_W = pickle.load(fp)
                return
        self.sigma = sigma
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.iterations = iterations
        self.network = network
        self.network_T = network_T
        self.D = D
        self.N, self.M, _ = np.max(train_data, axis=0)
        self.N, self.M = int(self.N), int(self.M)
        self.train_data = train_data
        self.user_list, self.item_list = utils.data_to_list(self.N, self.M, self.train_data)

    def init(self):
        '''
            initialized mean and cov of
            Qs uniformly in [0, 1].
        '''
        self.mean_U = np.random.rand(self.N+1, self.D)
        self.cov_U = np.random.rand(self.N+1, self.D, self.D)
        self.mean_V = np.random.rand(self.M+1, self.D)
        self.cov_V = np.random.rand(self.M+1, self.D, self.D)
        self.mean_W = [[np.random.rand() for i1 in range(len(self.network[i]))] for i in range(self.N+1)]
        self.var_W = [[np.random.rand() for i1 in range(len(self.network[i]))] for i in range(self.N+1)]

    def test(self, data):
        '''
            return RMSE of model on validation data
        '''
        rmse = 0.
        for i, j, score in data:
            i, j = int(i), int(j)
            pred = (self.E_U(i).T.dot(self.E_V(j)))[0,0]
            for i1 in self.network[i]:
                pred += (self.E_W(i, i1) * self.E_U(i1).dot(self.E_V(j)))[0,0]
            rmse += (pred - score)**2
        rmse /= data.shape[0]
        rmse = np.sqrt(rmse)
        return rmse

    def predict(self, data):
        '''
            return prediction for given data.
            data -> 2d array (user, item)
        '''
        preds = []
        for i, j in data:
            i, j = int(i), int(j)
            pred = (self.E_U(i).T.dot(self.E_V(j)))[0,0]
            for i1 in self.network[i]:
                pred += (self.E_W(i, i1) * self.E_U(i1).dot(self.E_V(j)))[0,0]
            preds.append(pred)
        return np.array(pred)

    def train(self):
        self.init()
        rmse = self.test(self.train_data)
        rmse_list = [rmse]
        print('iteration', 0, ' RMSE:', rmse)
        for k in range(1, self.iterations+1):
            for i in range(1, self.N+1):
                for i1 in self.network[i]:
                    self.update_W(i, i1)
            for i in range(1, self.N+1):
                self.update_U(i)
            for j in range(1, self.M+1):
                self.update_V(j)
            rmse = self.test(self.train_data)
            print('iteration', k, ' RMSE:', rmse)
            rmse_list.append(rmse)
        if not os.path.exists('model/'):
            os.makedirs('model/')
        np.save('model/E_U.npy', self.mean_U)
        np.save('model/E_V.npy', self.mean_V)
        with open('model/E_W.txt', 'wb') as fp:
            pickle.dump(self.mean_W, fp)
        return rmse_list

    def update_U(self, i):
        '''
            update mean and cov of Q(Ui)
        '''
        cov_inv = np.eye(self.D) / self.sigma_u
        mean = 0
        for j, score in self.user_list[i]:
            E_VVJ = self.E_VV(j)
            cov_inv += E_VVJ / self.sigma
            mean += (score * self.E_V(j) - E_VVJ.dot(self.E_S(i).T)) / self.sigma

        for i2 in self.network_T[i]:
            for j, score in self.user_list[i2]:
                E_VVJ = self.E_VV(j)
                cov_inv += (self.E_WW(i2, i) * E_VVJ) / self.sigma
                mean += self.E_W(i2, i) * \
                    (score * self.E_V(j) - E_VVJ.dot(self.E_U(i2) + self.E_S_(i2, i).T))
        cov = inv(cov_inv)
        mean = cov.dot(mean)
        self.mean_U[i, :] = mean
        self.cov_U[i, :, :] = cov

    def update_V(self, j):
        '''
            update mean and cov of Q(Vj)
        '''
        cov_inv = np.eye(self.D) / self.sigma_v
        mean = 0
        for i, score in self.item_list[j]:
            E_Si = self.E_S(i)
            cov_inv += (self.E_UU(i)+ 2 * self.E_U(i).dot(E_Si) +
                        self.E_SS(i)) / self.sigma
            mean += score * (self.E_U(i) + E_Si.T) / self.sigma
        cov = inv(cov_inv)
        mean = cov.dot(mean)
        self.mean_V[j, :] = mean
        self.cov_V[j, :, :] = cov

    def update_W(self, i, i1):
        '''
            update mean and cov of Q(Wii')
        '''
        rho = 1. / self.sigma_w
        mean = 0
        for j, score in self.user_list[i]:
            EUi1 = self.E_U(i1)
            EUi = self.E_U(i)
            E_VVJ = self.E_VV(j)
            rho += (self.E_UVVU(i1, j)) / self.sigma
            mean += (EUi1.T.dot(self.E_V(j)) * score -
                    EUi.T.dot(E_VVJ.dot(self.E_U(i1))) -
                    EUi1.T.dot(E_VVJ.dot(self.E_S_(i, i1)))) / self.sigma

        var = 1. / rho
        mean = var * mean
        idx = self.network[i].index(i1)
        self.mean_W[i][idx] = mean
        self.var_W[i][idx] = var

    def E_U(self, i):
        '''
            return EQ[Ui]
        '''
        return self.mean_U[i].reshape((self.D, 1))

    def E_UU(self, i):
        '''
            return EQ[UiUi^T]
        '''
        m_u_i = self.mean_U[i].reshape((self.D, 1))
        return (self.cov_U[i, :, :] + m_u_i.dot(m_u_i.T))

    def E_V(self, j):
        '''
            return EQ[Vj]
        '''
        return self.mean_V[j].reshape((self.D, 1))

    def E_VV(self, j):
        '''
            return E[VjVj^T]
        '''
        m_v_j = self.mean_V[j].reshape((self.D, 1))
        return (self.cov_V[j, :, :] + m_v_j.dot(m_v_j.T))

    def E_W(self, i, i1):
        '''
            return EQ[Wii']
        '''
        idx = self.network[i].index(i1)
        return self.mean_W[i][idx]

    def E_WW(self, i, i1):
        '''
            return EQ[Wii'Wii'^T]
        '''
        idx = self.network[i].index(i1)
        return self.var_W[i][idx] + self.mean_W[i][idx]**2

    def E_S(self, i):
        '''
            return EQ[Si]
        '''
        ret = np.zeros((self.D, 1))
        for i1 in self.network[i]:
            ret += self.E_W(i, i1) * self.E_U(i1)
        return ret

    def E_S_(self, i, k):
        '''
            return EQ[Si^(-k)]
        '''
        return self.E_S(i) - self.E_W(i, k) * self.E_U(k)

    def E_SS(self, i):
        '''
            return EQ[Si^TSi]
        '''
        ret = 0
        for i1 in self.network[i]:
            ret += self.E_WW(i, i1) * self.E_UU(i1)
            for i2 in self.network[i]:
                if i1 != i2:
                    ret += self.E_W(i, i1) * self.E_W(i, i2) * self.E_U(i1) * self.E_U(i2).T
        return ret

    def E_UVVU(self, i, j):
        '''
            return EQ[Ui^TVjVj^TUi]
        '''
        e = np.ones((self.D, 1))
        tmp1 = e.T.dot(self.E_UU(i).dot(e))
        tmp2 = e.T.dot(self.E_VV(j).dot(e))
        return tmp1 * tmp2
