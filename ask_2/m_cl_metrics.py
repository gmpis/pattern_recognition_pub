from itertools import combinations

m_debug = False

def print_cl_freq(num_cl, cl_Array):
    freq_Arr =[]

    # init
    for i in range(num_cl):
        freq_Arr.append(0)

    for pr in cl_Array:
        freq_Arr[pr] += 1

    # print
    for i in range(num_cl):
        print(i, ":", freq_Arr[i])


def cl_to_inst_index(num_cl, pred_Arr):
    # maps cluster_id and the indexes of all the instances belonging to that cluster
    out = {}

    # init
    for i in range(num_cl):
        out[i] = []

    # fill
    for d in range(len(pred_Arr)):
        cl_id = pred_Arr[d]
        out[cl_id].append(d)

    return out


def class_in_cluster_counter(num_cl, cl_inst_indx, real_labels):
    out = {}           # cluster_index:class_mapped_to

    # init
    for i in range(num_cl):
        out[i] = [] # TODO maybe extra

    for cluster_id in range(num_cl):
        class_counter = [0, 0]

        ind_arr = cl_inst_indx[cluster_id]   # contains the ids of all the instances that belongs to that cluster

        for inst_id in ind_arr:
            real_class = real_labels[inst_id]

            if real_class == 0:
                class_counter[0] += 1

            elif real_class == 1:
                class_counter[1] += 1

            else:
                print("Read unexpected class!! returning...")
                return None

        out[cluster_id] = class_counter

    return out


def class_y_incl(w_class, cntr): #uses prev method to return appearance array
    out = []
    #prev =  class_in_cluster_counter(num_cl, cl_inst_indx, real_labels)

    for i in range(len(cntr)):
        out.append(cntr[i][w_class])

    return out


def decide_cluster_categ(num_cl, m_distrb):
    # cluster_index:class_mapped_to
    # m_distrb = class_in_cluster_distr(num_cl, cl_pred_map, real_labels)
    out = {}

    for cluster_id in range(num_cl):

        # print("Working on cluster: ",cluster_id)
        class_counter = m_distrb[cluster_id]

        if class_counter[0] > class_counter[1]:
            out[cluster_id] = 0

        elif class_counter[0] < class_counter[1]:
            out[cluster_id] = 1

        else:
            print("Same number of 0 and 1 class in this cluster, choosing 0...") # TODO maybe change
            out[cluster_id] = 0

    return out


def calc_purity(num_cl, N_sampl, m_class_distr): # real_labels is last cl on dset
    # perfect clustering purity :1, bad close to: 0

    sum_maxes = 0

    for cluster_id in range(num_cl):
        cur_dist = m_class_distr[cluster_id]

        cur_max = max(cur_dist)
        sum_maxes += cur_max

    return sum_maxes / N_sampl


def calc_f_measure(num_cl, cl_ind_map, categ_map, real_labels, m_counter):

    total_f = 0

    for cluster_id in range(num_cl):
        # print("Working on cluster: ",cluster_id)

        cluster_f = 0   # reset f measure for this cluster

        True_Pos = 0    # 2 similar docs to same cluster
        False_Pos = 0   # 2 dissimilar docs to same cluster

        False_Neg = 0   # 2 similar docs to different clusters
        True_Neg = 0    # extra, not used here


        ind_arr = cl_ind_map[cluster_id]
        cluster_categ = categ_map[cluster_id]

        # print("EL: ", ind_arr) # TODO debug print

        same_cl_comb = combinations(ind_arr, 2) # combinatons of elements in same cluster
        for ci in list(same_cl_comb): # ci is type typle
            # print("Comb", ci)
            # print(type(ci))

            # f_el, s_el in same cluster
            f_el = ci[0]
            s_el = ci[1]

            if real_labels[f_el] == real_labels[s_el]:
                True_Pos += 1
            else:
                False_Pos += 1

        # calc False_Neg: 2 similar docs to different clusters
        for c_loop in range(2):
            tmp_lst = class_y_incl(c_loop, m_counter)

            combTw = combinations(tmp_lst, 2)  # combinatons of elements in same cluster
            for citm in list(combTw):  # ci is type typle
               False_Neg += citm[0] * citm[1]



        # for inst_id in ind_arr:
        #     real_class = real_labels[inst_id]
        #
        #     if cluster_categ == 1 and real_class == 1:
        #         True_Pos += 1



        # calc f measure based on pdf
        m_precision = True_Pos / (True_Pos + False_Pos)
        m_recall = True_Pos / (True_Pos + False_Neg)

        if(m_debug):
            print("On cluster [", cluster_id, "]::", "TP: ", True_Pos, "FP: ", False_Pos, "FN: ", False_Neg)
            print("\t\t::", "precision: ", m_precision, "m_recall: ", m_recall,"\n")

        # m_a_const = 1
        # m_arithm = 1 + m_a_const
        # m_par = (1 / m_precision) + (m_a_const / m_recall)

        # f1 score can be simplified more
        m_arithm = 2 * m_precision * m_recall
        m_par = m_precision + m_recall
        cluster_f = m_arithm / m_par

        total_f += cluster_f
    return total_f


def m_calc_all_meas(k_cl, res, r_categ):
    out = {}

    N_sampl = len(res)
    cl_ind_map = cl_to_inst_index(k_cl, res)
    m_distrb = class_in_cluster_counter(k_cl, cl_ind_map, r_categ)
    categ_map = decide_cluster_categ(k_cl, m_distrb)

    out["purity"] = calc_purity(k_cl, N_sampl, m_distrb)
    out["f_measure"] = calc_f_measure(k_cl, cl_ind_map, categ_map, r_categ, m_distrb)

    return out