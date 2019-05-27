import csv


def m_load_dset(m_set):

    if m_set == "spam":
        print("Loading spam dataset...")
        path_spm_data = "../raw_data/spambase/spambase.data"
        # valid_features_len  = 58      # including class, last element 0,1
        # valid_lines_num     = 4601    # expected number of lines to read
        f_data = m_custom_loader(path_spm_data, 58, 0, False)
    else:
        print("Loading occupancy dataset...")
        # path_occ_data = "./raw_data/occupancy_data/datatest.txt"
        path_occ_data = "../raw_data/occupancy_data/datatraining.txt"
        # valid_features_len  = 8           # including class, last element 0,1
        # valid_lines_num     = 8143        # expected number of lines to read
        f_data = m_custom_loader(path_occ_data, 8, 2, True)  # skip first 2 col, and first row

    return f_data


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


# debug TODO remove
#m_load_dset("occ")