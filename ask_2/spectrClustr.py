from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

from scipy.sparse import csgraph, linalg
# from scipy.linalg import eigvalsh
from m_cl_loader import m_load_dset
from m_cl_metrics import  m_calc_all_meas, print_cl_freq

us_dSet_name = "occ"       # opt: "occ" , "spam"
us_data = m_load_dset(us_dSet_name)

K = 2  # number of clusters opt: 2 , 4 , 8


def m_spctrClustr(k_cl = 2): # k means



    # sc = SpectralClustering(n_clusters=k_cl, affinity='nearest_neighbors', assign_labels="kmeans")
    sc = SpectralClustering(n_clusters=k_cl, affinity='rbf', assign_labels="discretize") # less sensitive to init


    m_scaler = StandardScaler()  # better results,
    train_dat_Arr = m_scaler.fit_transform(us_data[0])

    sc.fit(train_dat_Arr)
    cl_res = sc.labels_


    # print eval measures
    r_categ = us_data[1]  # array of int 0 or 1, get from dataset

    m_metr = m_calc_all_meas(k_cl, cl_res, r_categ)
    pur_int = m_metr["purity"]
    fm_int = m_metr["f_measure"]
    print("[specClust]", "K:", k_cl, "Purity: ", pur_int, "F-measure: ", fm_int)


    # calc eigvals for diagram
    print("Working on eigenvals: ")
    # availiable only after calling fit ,
    #aff_mtrx = sc.affinity_matrix_  # <class 'numpy.ndarray'>, shape(n_samples x n_samples)

    aff_mtrx = kneighbors_graph(train_dat_Arr, 2, mode="distance", metric="minkowski", p=2, metric_params=None, include_self=False)
    L = csgraph.laplacian(aff_mtrx, normed=False)  # returns the Laplacian matrix of a directed graph
    # untill here fast, same shape as aff_matr


    eigval, eighvec = linalg.eigsh(L, k = k_cl, which="LM", maxiter=50000) # after many minutes
    # eigval, eighvec = np.linalg.eig(L)
    # eigval = eigvalsh(L)
    print("eigv", eigval)


m_spctrClustr(K)
