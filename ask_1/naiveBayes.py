from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
from time import time
from m_loader import m_load_dset, m_folds_split
#import numpy

us_folds = m_load_dset("spam")     # opt: "occ" , "spam"

def m_naivBayes(folds):

    # use to calc avg
    num_folds = len(folds)  # num of folds == num of tests
    sum_acc = 0         # sum of accuracy from testing each fold
    sum_f1_score = 0    # sum of f1 score from testing each fold

    sum_tr_time = 0

    #create matrix based on folds
    for tst_num in range(num_folds):
        #print("Test: ", tst_num)

        #init clf
        clf = GaussianNB()

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

        # here have correct arrays calculate metrics
        # built-in funcs provide marginally better results, e^16 diff, so use those
        sum_acc += accuracy_score(cl_res, test_dat_Lb_Arr)
        sum_f1_score += f1_score(cl_res, test_dat_Lb_Arr)


        # other, class 'numpy.ndarray', len 2 [[57],[57]] == 2x57 ,for spam
        #sigma_ variance of each feature per class
        #theta_ : mean of each feature per class
        #print("Std z: ",clf.sigma_[0], "Mean z: ", clf.theta_[0]) # maybe numpy.sqrt(clf.sigma_[0])



    # here all tests have finished calc avg values
    avg_accuracy = sum_acc / num_folds
    avg_f1_score = sum_f1_score / num_folds

    print("[nvBayes]: ", "avg_accuracy: ", avg_accuracy, " avg_f1_score: ", avg_f1_score)

    #maybe extra
    avg_tr_time = sum_tr_time / num_folds
    avg_tr_time = round(avg_tr_time, 4)     # round to 4 digits
    #print("avg_trn_t:", avg_tr_time, "s")

m_naivBayes(us_folds)

# spam dataset:
# [nvBayes]:  avg_accuracy:  0.8222103642718785  avg_f1_score:  0.8100021880133197
# [nvBayes]:  avg_accuracy:  0.8206942880267578  avg_f1_score:  0.8084231134156203
# [nvBayes]:  avg_accuracy:  0.8215572472295319  avg_f1_score:  0.8088975623607404
# [nvBayes]:  avg_accuracy:  0.8209291600991785  avg_f1_score:  0.8084681450540682
