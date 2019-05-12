import numpy as np
import sys
import pandas as pd

from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, sigmoid_kernel


class MMD:
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        # self.X_test = X_test
        # self.y_test = y_test
        self.gamma = 0.00000000026
        # self.kernel = rbf_kernel(self.X, gamma=self.gamma)

    def acquire_prototypes_and_criticisms(self, num_prototypes, num_criticisms):
        p_idx, c_idx = self.acquire_prototypes_and_criticisms_idx(num_prototypes, num_criticisms)

        return self.X[p_idx, :], self.X[c_idx, :]


    def acquire_prototypes_and_criticisms_idx(self, num_prototypes, num_criticisms):

        p_size = num_prototypes
        c_size = num_criticisms


        num_prototypes = int(2 * num_prototypes)
        if num_prototypes > len(self.X):
            num_prototypes = len(self.X)

        num_criticisms = int(2 * num_criticisms)
        if num_criticisms > len(self.X):
            num_criticisms = len(self.X)

        self.calculate_kernel_individual()

        p_selected = self.greedy_select_protos(self.kernel, np.array(range(np.shape(self.kernel)[0])), num_prototypes)
        p_selected = self.__filter_selected(p_selected, p_size, 0)

        p_selectedy = self.y[p_selected]
        p_sortedindx = np.argsort(p_selectedy)
        # p_X = self.X[p_selected[p_sortedindx], :]

        c_selected = self.select_criticism_regularized(self.kernel, p_selected, num_criticisms, is_K_sparse=False)
        c_selected = self.__filter_selected(c_selected, c_size, 1)

        c_selectedy = self.y[c_selected]
        c_sortedindx = np.argsort(c_selectedy)
        # c_X = self.X[c_selected[c_sortedindx], :]

        return p_selected[p_sortedindx], c_selected[c_sortedindx]

    ##############################################################################################################################
    # Function choose m of all rows by MMD as per kernelfunc
    # ARGS:
    # K : kernel matrix
    # candidate_indices : array of potential choices for selections, returned values are chosen from these  indices
    # m: number of selections to be made
    # is_K_sparse:  True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix.
    # RETURNS: subset of candidate_indices which are selected as prototypes
    ##############################################################################################################################
    def greedy_select_protos(self, K, candidate_indices, m, is_K_sparse=False):

        if len(candidate_indices) != np.shape(K)[0]:
            K = K[:, candidate_indices][candidate_indices, :]

        n = len(candidate_indices)

        # colsum = np.array(K.sum(0)).ravel() # same as rowsum
        if is_K_sparse:
            colsum = 2 * np.array(K.sum(0)).ravel() / n
        else:
            colsum = 2 * np.sum(K, axis=0) / n

        selected = np.array([], dtype=int)
        value = np.array([])
        for i in range(m):
            maxx = -sys.float_info.max
            argmax = -1
            candidates = np.setdiff1d(range(n), selected)

            s1array = colsum[candidates]
            if len(selected) > 0:
                temp = K[selected, :][:, candidates]
                if is_K_sparse:
                    # s2array = temp.sum(0) *2
                    s2array = temp.sum(0) * 2 + K.diagonal()[candidates]

                else:
                    s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]

                s2array = s2array / (len(selected) + 1)

                s1array = s1array - s2array

            else:
                if is_K_sparse:
                    s1array = s1array - (np.abs(K.diagonal()[candidates]))
                else:
                    s1array = s1array - (np.abs(np.diagonal(K)[candidates]))

            argmax = candidates[np.argmax(s1array)]
            # print "max %f" %np.max(s1array)

            selected = np.append(selected, argmax)
            # value = np.append(value,maxx)
            KK = K[selected, :][:, selected]
            if is_K_sparse:
                KK = KK.todense()

            inverse_of_prev_selected = np.linalg.pinv(KK)  # shortcut

        return candidate_indices[selected]

    ##############################################################################################################################
    # function to select criticisms
    # ARGS:
    # K: Kernel matrix
    # selectedprotos: prototypes already selected
    # m : number of criticisms to be selected
    # reg: regularizer type.
    # is_K_sparse:  True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix.
    # RETURNS: indices selected as criticisms
    ##############################################################################################################################
    def select_criticism_regularized(self, K, selectedprotos, m, reg='logdet', is_K_sparse=False):

        n = np.shape(K)[0]
        if reg in ['None', 'logdet', 'iterative']:
            pass
        else:
            exit(1)

        selected = np.array([], dtype=int)
        candidates2 = np.setdiff1d(range(n), selectedprotos)
        inverse_of_prev_selected = None  # should be a matrix

        if is_K_sparse:
            colsum = np.array(K.sum(0)).ravel() / n
        else:
            colsum = np.sum(K, axis=0) / n

        for i in range(m):
            candidates = np.setdiff1d(candidates2, selected)

            s1array = colsum[candidates]

            temp = K[selectedprotos, :][:, candidates]
            if is_K_sparse:
                s2array = temp.sum(0)
            else:
                s2array = np.sum(temp, axis=0)

            s2array = s2array / (len(selectedprotos))

            s1array = np.abs(s1array - s2array)
            if reg == 'logdet':
                if inverse_of_prev_selected is not None:  # first call has been made already
                    temp = K[selected, :][:, candidates]
                    if is_K_sparse:
                        temp2 = temp.transpose().dot(inverse_of_prev_selected)
                        regularizer = temp.transpose().multiply(temp2)
                        regcolsum = regularizer.sum(1).ravel()  # np.sum(regularizer, axis=0)
                        regularizer = np.abs(K.diagonal()[candidates] - regcolsum)

                    else:
                        # hadamard product
                        temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                        regularizer = temp2 * temp
                        regcolsum = np.sum(regularizer, axis=0)
                        regularizer = np.log(np.abs(np.diagonal(K)[candidates] - regcolsum))
                    s1array = s1array + regularizer
                else:
                    if is_K_sparse:
                        s1array = s1array - np.log(np.abs(K.diagonal()[candidates]))
                    else:
                        s1array = s1array - np.log(np.abs(np.diagonal(K)[candidates]))
            argmax = candidates[np.argmax(s1array)]

            selected = np.append(selected, argmax)
            if reg == 'logdet':
                KK = K[selected, :][:, selected]
                if is_K_sparse:
                    KK = KK.todense()

                inverse_of_prev_selected = np.linalg.pinv(KK)  # shortcut
            if reg == 'iterative':
                selectedprotos = np.append(selectedprotos, argmax)

        return selected

    def calculate_kernel_individual(self, g=None):
        touseg = g
        if touseg is None:
            touseg = self.gamma
        if touseg is None:
            exit(1)
        self.kernel = np.zeros((np.shape(self.X)[0], np.shape(self.X)[0]))
        sortind = np.argsort(self.y)
        self.X = self.X[sortind, :]
        self.y = self.y[sortind]

        for i in np.arange(2):
            j = i
            ind = np.where(self.y == j)[0]
            startind = np.min(ind)
            endind = np.max(ind) + 1
            self.kernel[startind:endind, startind:endind] = sigmoid_kernel(self.X[startind:endind, :], gamma=self.gamma)

    def __filter_selected(self, selected, size, to_filter):
        if size > len(selected):
            return selected

        indeces = np.where(self.y[selected] == to_filter)[0]
        indeces = indeces[:len(selected) - size]
        selected = np.delete(selected, indeces)

        if len(selected) > size:
            selected = selected[:size]

        return selected


# X = pd.read_csv('../loan_model_X_train_data.csv')
# y = pd.read_csv('../loan_model_y_train_data.csv')
#
# mmd = MMD(X.values, y.values.reshape(1, -1)[0])
#
# p_X, c_X = mmd.acquire_prototypes_and_criticisms(5, 5)
#
# a = 10
