from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from m_loader import m_load_dset, m_folds_split
#from m_metrics import calc_metrics
from time import time


us_folds = m_load_dset("spam")      # opt: "occ" , "spam"


def m_kNNeighbor(folds, num_neigh): # k param, change to see diff in performance

    # params for KNeighborsClassifier
    m_weights = 'distance'  # opt:'uniform', 'distance'
    m_algo = 'brute'        # opt: 'auto' , 'brute' :: prefer auto, because spam has many features

    # use to calc avg
    num_folds = len(folds)  # num of folds == num of tests
    sum_acc = 0             # sum of accuracy from testing each fold
    sum_f1_score = 0        # sum of f1 score from testing each fold

    sum_tr_time = 0

    #create matrix based on folds
    for tst_num in range(num_folds):
        # print("Test: ", tst_num)

        #init clf
        clf = KNeighborsClassifier(num_neigh, weights=m_weights, algorithm=m_algo)

        spl_res = m_folds_split(folds, num_folds, tst_num)
        train_dat_Arr = spl_res["train_dat_Arr"]
        train_dat_Lb_Arr = spl_res["train_dat_Lb_Arr"]

        test_dat_Arr = spl_res["test_dat_Arr"]
        test_dat_Lb_Arr = spl_res["test_dat_Lb_Arr"]



        # here train_dat_Arr & train_dat_Lb_Arr populated correctly, so can train now
        perf_st_time = time()
        clf.fit(train_dat_Arr, train_dat_Lb_Arr)

        # not very accurate time measurement, python has better funcs for profiling
        perf_diff = round(time() - perf_st_time, 4)  # diff in sec
        sum_tr_time += perf_diff

        cl_res = clf.predict(test_dat_Arr)

        # here have correct arrays calculate metrics built-in funcs provide marginally better results, e^16 diff, so use those
        sum_acc += accuracy_score(cl_res, test_dat_Lb_Arr)
        sum_f1_score += f1_score(cl_res, test_dat_Lb_Arr)

        # OLD CODE for comp
        # m_mtr = calc_metrics(cl_res, test_dat_Lb_Arr)
        # f1_score(cl_res, test_dat_Lb_Arr) -  m_mtr["f1_score"] #:: m_mtr["accuracy"]


    # here all tests have finished calc avg values
    avg_accuracy = sum_acc / num_folds
    avg_f1_score = sum_f1_score / num_folds
    print("[kNN]: ", "k: ", num_neigh, " avg_accuracy: ", avg_accuracy, " avg_f1_score: ", avg_f1_score)

    avg_tr_time = sum_tr_time / num_folds
    avg_tr_time = round(avg_tr_time,4) # round to 4 digits
    #print("\t \t \t avg_trn_t:", avg_tr_time, "s")




#for i in range(1, 50, 3):
# for i in range(1, 15):
#     m_kNNeighbor(us_folds, i)


# very good permfomance k == 4
m_kNNeighbor(us_folds, 4)