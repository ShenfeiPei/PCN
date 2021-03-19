import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

import funs as Funs
import funs_metric as Mfuns
from PCN import PCN


knn = 33
X, y_true, N, dim, c_true = Funs.load_mat("toydata/Agg.mat")

D_full = EuDist2(X, X, squared=True)
np.fill_diagonal(D_full, -1)
NN_full = np.argsort(D_full, axis=1)
np.fill_diagonal(D_full, 0)

NN = NN_full[:, 1:(knn+1)]
NND = Funs.matrix_index_take(D_full, NN)

for i in range(N):
    tmp_ind = np.lexsort((NN[i, :], NND[i, :]))
    NN[i, :] = NN[i, tmp_ind]

model = PCN(NN, NND)
y_pred = model.cluster()

pre = Mfuns.precision(y_true=y_true, y_pred=y_pred)
rec = Mfuns.recall(y_true=y_true, y_pred=y_pred)
f1 = 2 * pre * rec / (pre + rec)

print("{:.3f}".format(f1))
