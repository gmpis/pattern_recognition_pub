from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from m_cl_loader import m_load_dset
from m_cl_metrics import  m_calc_all_meas, print_cl_freq

us_dSet_name = "spam"       # opt: "occ" , "spam"
us_data = m_load_dset(us_dSet_name)

K = 2  # number of clusters opt: 2 , 4 , 8


def m_agglomHr(k_cl = 2):
    # best results ward [0 : 2290,  1 : 5853]
    m_lnkg = "ward" # opt: "complete", "single", "average", "ward"
    agg = AgglomerativeClustering(n_clusters=k_cl, affinity="euclidean", linkage=m_lnkg)

    m_scaler = StandardScaler()  # better results,
    train_dat_Arr = m_scaler.fit_transform(us_data[0])

    agg.fit(train_dat_Arr)
    cl_res = agg.labels_

    #cl_res = agg.fit_predict(train_dat_Arr)


    # TODO: debug items, remove later
    # print(cl_res)
    # print_cl_freq(k_cl, cl_res)

    # print eval measures
    r_categ = us_data[1]  # array of int 0 or 1, get from dataset

    m_metr = m_calc_all_meas(k_cl, cl_res, r_categ)
    pur_int = m_metr["purity"]
    fm_int = m_metr["f_measure"]
    print("[agglom]", "K:", k_cl, "Purity: ", pur_int, "F-measure: ", fm_int)

m_agglomHr(K)
