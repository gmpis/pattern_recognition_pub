from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from m_loader import m_load_dset, m_folds_split

# Run params passed as arguments to m_neural_net_run
us_dSet_name = "spam"       # opt: "occ" , "spam", used for importing and allowing Scaler to run
us_folds = m_load_dset(us_dSet_name)

us_hid_layer_sz = (150,)  # 1 Layer (9,) OR 2 Layers: (4, 1)
#invs gradually decreases the learning rate at each time step
us_learn_rt = 'invscaling'  # opt: 'constant', 'invscaling', 'adaptive'
us_rt_init = 0.075
us_max_itr = 600         # 450 not enough


#us_learn_rt = 'adaptive'
#us_rt_init = 0.001

def m_neural_net_run(m_d_set, folds, m_hid_layer_sz, m_learn_rt, m_rt_init, m_max_itr):

    print("Neural params: L:" , m_hid_layer_sz, " LRt: ", m_learn_rt, "RT_INIT", m_rt_init, "MAX_ITER", m_max_itr)
    # neural network params
    # m_hid_layer_sz    : (150,)     # 1 Layer (9,) OR  2 Layers: (4, 1)
    # m_learn_rt        : 'adaptive' # opt: 'constant', 'invscaling', 'adaptive'
    # m_rt_init         : 0.001
    # m_max_itr         : prevent ConvergenceWarning: Stochastic Optimizer: Maximum iterations (450) reached and the optimization hasn't converged yet
    # activation        :"logistic" logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)
    # solver sgd        : stochastic gradient descent, missing gradient descent

    # use to calc avg
    num_folds = len(folds)  # num of folds == num of tests
    sum_acc = 0         # sum of accuracy from testing each fold
    sum_f1_score = 0    # sum of f1 score from testing each fold

    for tst_num in range(num_folds):
        print("Test: ", tst_num)
        #init clf
        clf = MLPClassifier(hidden_layer_sizes=m_hid_layer_sz, activation="logistic", solver="sgd", learning_rate_init=m_rt_init, learning_rate=m_learn_rt, max_iter=m_max_itr)# alpha=0.0001, tol=0.0001 random_state=1 )

        spl_res = m_folds_split(folds, num_folds, tst_num)

        train_dat_Arr = spl_res["train_dat_Arr"]
        train_dat_Lb_Arr = spl_res["train_dat_Lb_Arr"]

        test_dat_Arr = spl_res["test_dat_Arr"]
        test_dat_Lb_Arr = spl_res["test_dat_Lb_Arr"]

        # scale each feature for spam dataset, LEAVE HERE
        if m_d_set == "spam":
            m_scaler = StandardScaler() #better results, DONT USE MinMaxScaler(), cause no true samples
            m_scaler.fit(train_dat_Arr)
            train_dat_Arr = m_scaler.transform(train_dat_Arr)
            test_dat_Arr = m_scaler.transform(test_dat_Arr)

        # here finished scaling continue normall
        clf.fit(train_dat_Arr, train_dat_Lb_Arr)
        cl_res = clf.predict(test_dat_Arr)


        # print(cl_res) #debug TODO remove

        # here have correct arrays calculate metrics
        sum_acc += accuracy_score(cl_res, test_dat_Lb_Arr)
        sum_f1_score += f1_score(cl_res, test_dat_Lb_Arr)


    avg_accuracy = sum_acc / num_folds
    avg_f1_score = sum_f1_score / num_folds

    print("[NeuralNets]", "avg_accuracy: ", avg_accuracy, " avg_f1_score: ", avg_f1_score)



m_neural_net_run(us_dSet_name, us_folds, us_hid_layer_sz, us_learn_rt, us_rt_init, us_max_itr)

# avg_accuracy:  0.9237139075819583  avg_f1_score:  0.9005808109312575 with: adaptive, max iter 777, but very slow

# Neural params: L: (150,)  LRt:  invscaling RT_INIT 0.1 MAX_ITER 600:
# avg_accuracy:  0.9015427807766825  avg_f1_score:  0.8703818013220704
# avg_accuracy:  0.9030621477006239  avg_f1_score:  0.872266803900463