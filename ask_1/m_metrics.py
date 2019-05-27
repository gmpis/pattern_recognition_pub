def calc_metrics(alg_res_arr, real_label_arr):

    out = {"accuracy":0, "f1_score":0}

    if len(alg_res_arr) != len(real_label_arr):
        print(" score arrays length miss-match ")
        return None

    tr_pos = 0
    tr_neg = 0
    fal_pos = 0
    fal_neg = 0

    for cd_ind in range(len(alg_res_arr)):
        r = alg_res_arr[cd_ind]
        if r == 1:
            if real_label_arr[cd_ind] == 1:
                # true positive
                tr_pos += 1
            else:
                fal_pos += 1
        elif r == 0:
            if real_label_arr[cd_ind] == 0:
                # true negative
                tr_neg += 1
            else:
                fal_neg += 1

    out["accuracy"] = gen_accuracy(tr_pos, tr_neg, fal_pos, fal_neg)
    out["f1_score"] = gen_f1_score(tr_pos, tr_neg, fal_pos, fal_neg)

    return out


# internal helper functions
def gen_accuracy(tr_pos, tr_neg, fal_pos, fal_neg):
    total_p = tr_pos + fal_pos
    total_n = tr_neg + fal_neg
    return (tr_pos+tr_neg) / (total_p + total_n)


def gen_f1_score(tr_pos, tr_neg, fal_pos, fal_neg):
    d_tr_pos = 2*tr_pos
    return d_tr_pos / (d_tr_pos + fal_pos + fal_neg)

