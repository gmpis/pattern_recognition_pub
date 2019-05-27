from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from m_loader import m_load_dset, m_folds_split

us_dSet_name = "spam"       # opt: "occ" , "spam", used for importing and allowing Scaler to run
us_folds = m_load_dset(us_dSet_name)
us_sel_kernel = "linear"    # opt: "rbf" , "linear"
us_gamm = "auto"            # Kernel coefficient for ‘rbf’
us_c_val = 1.0              # Penalty parameter C of the error term, decision boundary and outliers

def m_svm_run(m_d_set, folds, m_kernel, m_gamm, m_C_val):

    # use to calc avg
    num_folds = len(folds)  # num of folds == num of tests
    sum_acc = 0         # sum of accuracy from testing each fold
    sum_f1_score = 0    # sum of f1 score from testing each fold

    #create matrix based on folds
    for tst_num in range(num_folds):
        print("Test: ", tst_num)

        # init clf
        if m_kernel == "rbf":
            clf = SVC(kernel="rbf", gamma=m_gamm, C= m_C_val)
        else:
            clf = SVC(kernel="linear", C= m_C_val)

        spl_res = m_folds_split(folds, num_folds, tst_num)

        train_dat_Arr = spl_res["train_dat_Arr"]
        train_dat_Lb_Arr = spl_res["train_dat_Lb_Arr"]

        test_dat_Arr = spl_res["test_dat_Arr"]
        test_dat_Lb_Arr = spl_res["test_dat_Lb_Arr"]

        if m_d_set == "spam":
            m_scaler = StandardScaler()  # better results, than MinMaxScaler
            m_scaler.fit(train_dat_Arr)

            train_dat_Arr = m_scaler.transform(train_dat_Arr)
            test_dat_Arr = m_scaler.transform(test_dat_Arr)

        clf.fit(train_dat_Arr, train_dat_Lb_Arr)
        cl_res = clf.predict(test_dat_Arr)

        # here have correct arrays calculate metrics
        # built-in funcs provide marginally better results, e^16 diff, so use those
        sum_acc += accuracy_score(cl_res, test_dat_Lb_Arr)
        sum_f1_score += f1_score(cl_res, test_dat_Lb_Arr)

    avg_accuracy = sum_acc / num_folds
    avg_f1_score = sum_f1_score / num_folds

    print("[SVM]", " avg_accuracy: ", avg_accuracy, " avg_f1_score: ", avg_f1_score)




m_svm_run(us_dSet_name, us_folds, us_sel_kernel, us_gamm, us_c_val)


# StandardScaler()  : [SVM]  avg_accuracy:  0.9282758342944417  avg_f1_score:  0.9076301157216147
# StandardScaler()  : [SVM]  avg_accuracy:  0.9278344374521372  avg_f1_score:  0.9070567074434643
# StandardScaler()  : [SVM]  avg_accuracy:  0.9289237230306628  avg_f1_score:  0.908282488805356

# MinMaxScaler()    : [SVM]  avg_accuracy:  0.9063235196517121  avg_f1_score:  0.8751266171542731
# MinMaxScaler()    : [SVM]  avg_accuracy:  0.9032903993751923  avg_f1_score:  0.8707541632518229
# MinMaxScaler()    : [SVM]  avg_accuracy:  0.9048050629811064  avg_f1_score:  0.872919311457004

# noScaler          : very very very slow, stopped it before getting an answer
