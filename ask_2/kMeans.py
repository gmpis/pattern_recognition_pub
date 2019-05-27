from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from m_cl_loader import m_load_dset
from m_cl_metrics import  m_calc_all_meas, print_cl_freq

us_dSet_name = "spam"       # opt: "occ" , "spam"
us_data = m_load_dset(us_dSet_name)

K = 2  # number of clusters opt: 2 , 4 , 8



def m_kMeans(k_cl = 2): # k means

    kmn = KMeans(n_clusters=k_cl, n_init=10)

    m_scaler = StandardScaler()  # better results,
    train_dat_Arr = m_scaler.fit_transform(us_data[0])

    # cl_res = kmn.fit_predict(train_dat_Arr)

    kmn.fit(train_dat_Arr)
    cl_res = kmn.labels_

    r_categ = us_data[1] # array of int 0 or 1, get from dataset

    
    m_metr = m_calc_all_meas(k_cl, cl_res, r_categ)
    pur_int = m_metr["purity"]
    fm_int = m_metr["f_measure"]

    print("[kMeans]", "K:", k_cl, "Purity: ", pur_int, "F-measure: ", fm_int)




m_kMeans(K)

