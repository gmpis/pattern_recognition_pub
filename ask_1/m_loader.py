import csv
from numpy import random as num_Rand


def m_load_dset(m_set):

    if m_set == "spam":
        print("Loading spam dataset...")
        path_spm_data = "./raw_data/spambase/spambase.data"
        # valid_features_len  = 58      # including class, last element 0,1
        # valid_lines_num     = 4601    # expected number of lines to read
        f_data = m_custom_loader(path_spm_data, 58, 0, False)
    else:
        print("Loading occupancy dataset...")
        # path_occ_data = "./raw_data/occupancy_data/datatest.txt"
        path_occ_data = "./raw_data/occupancy_data/datatraining.txt"
        # valid_features_len  = 8           # including class, last element 0,1
        # valid_lines_num     = 8143        # expected number of lines to read
        f_data = m_custom_loader(path_occ_data, 8, 2, True)  # skip first 2 col, and first row

    fld_arr = m_create_folds(f_data,10)
    return fld_arr


def m_custom_loader(m_file_path, n_features, skp_feat, has_header):
    out = [[], []]  # init to empty w len 2, contains data and class arrays
    out_data_arr = []
    out_class_arr = []


    f = open(m_file_path)
    r_lines_arr = csv.reader(f, delimiter=",")
    v_read_counter = 0   # number of valid lines read

    found_header = False # used with has header param

    for line in r_lines_arr :

        if has_header == True and found_header == False:
            found_header = True
            continue # skip first line with header

        line_len = len(line)
        if line_len != n_features:
            print("Found line with miss-match number of attributes /n", "Expected", n_features, "found: ", line_len)
            # print(line)
            return None

        tmp_arr = []
        strt_index = skp_feat           # allows to skip features in the beggining eg id, time
        cls_index = n_features-1
        tmp_cls = int(line[cls_index])  # always last element is class


        if (tmp_cls != 0 and tmp_cls != 1):
            print("Read invalid class \n  Exiting now ...")
            return None


        # print(line)
        # if(tmp_cls == 1 ):
        #     print("Class: Spam :(")
        # else:
        #     print("Class: Not spam.")


        for feat in line[strt_index:cls_index]:
            tmp_conv_el = float(feat)
            tmp_arr.append(tmp_conv_el)


        # if line is valid
        v_read_counter +=1
        out_data_arr.append(tmp_arr)
        out_class_arr.append(tmp_cls)

    print(">> Imported ", v_read_counter," lines from file: ", m_file_path)

    # have imported and normalized data, so populate out array
    out[0]=out_data_arr
    out[1]=out_class_arr
    return out


def m_create_folds(data, fold_numb=10):
    folds_Arr = [] # [fold1, fold2, ... ] len 10

    # init Array
    for i in range(fold_numb):
        # init each fold {data,labels}
        folds_Arr.append({"data": [], "labels": []})

    ft = data[0]
    lb = data[1]

    num_data = len(ft)

    if len(ft) != len(lb):
        print("Miss-match between arrays len, features len: ",len(ft), " labels len: ", len(lb), " \n Exiting now...")
        return None

    class_Zero = []
    count_Zero = 0

    class_One = []
    count_One = 0


    for item in range(num_data):
        if lb[item] == 0:
            class_Zero.append(ft[item])
            count_Zero +=1
        elif lb[item] == 1:
            class_One.append(ft[item])
            count_One +=1
        else:
            print("Unexpected ")
            return None


    print("Class 0: ", count_Zero,", Class 1: ", count_One, ", Total: ", count_One+count_Zero)


    # For class zero
    tmp_fold_ind = 0 # in which fold it belong
    tmp_visit_ord = num_Rand.permutation(count_Zero)
    for rand_ind in tmp_visit_ord:
        folds_Arr[tmp_fold_ind]["data"].append(class_Zero[rand_ind])
        folds_Arr[tmp_fold_ind]["labels"].append(0)
        tmp_fold_ind = (tmp_fold_ind + 1) % 10 # mod 10 possible val:0 9


    # For class one
    tmp_fold_ind = 0 # in which fold it belong
    tmp_visit_ord = num_Rand.permutation(count_One)
    for rand_ind in tmp_visit_ord:
        folds_Arr[tmp_fold_ind]["data"].append(class_One[rand_ind])
        folds_Arr[tmp_fold_ind]["labels"].append(1)
        tmp_fold_ind = (tmp_fold_ind + 1) % 10 # mod 10 possible val:0 9


    return folds_Arr


# debug function
def debug_folds(fld):
    for i in range(len(fld)):
        print("Fold: ", i)
        val = fld[i]
        print(val["data"])
        print(val["labels"])
        print("Dat Size: ", len(val["data"]))
        print("Labl Size: ", len(val["labels"]))
        print("------------")


# given folds and test index splits to training set and test set, used inside for loop
def m_folds_split(folds, num_folds, test_indx):
    out = {}

    test_dat_Arr = folds[test_indx]["data"]
    test_dat_Lb_Arr = folds[test_indx]["labels"]

    train_dat_Arr = []
    train_dat_Lb_Arr = []

    # use all other data as training set
    for f in range(num_folds):
        if f == test_indx:
            # training set index, skip this
            continue
        else:
            train_dat_Arr = train_dat_Arr + folds[f]["data"]
            train_dat_Lb_Arr = train_dat_Lb_Arr + folds[f]["labels"]

    out = {"test_dat_Arr":test_dat_Arr, "test_dat_Lb_Arr":test_dat_Lb_Arr, "train_dat_Arr":train_dat_Arr, "train_dat_Lb_Arr":train_dat_Lb_Arr}
    return out
