import numpy as np
import theano.tensor as T
import pdb
from keras.models import model_from_json
from tabulate import tabulate



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], np.asarray(b)[p]

#def shuffle_weights(model):
#    weights = model.get_weights()
#    #pdb.set_trace()
#    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
#    #pdb.set_trace()
#    model.set_weights(weights)

def round(arr):
    return np.round(np.array(arr, dtype=np.float64))

def compute_avg_seg_len(y_true):
    idx = np.where(y_true == 1)[0]
    seg_size, seg_count = 0.0, idx.shape[0]
    for i in range(seg_count):
        if i == seg_count - 1:
            seg_size += y_true.shape[0] - idx[i]
        else:
            seg_size += idx[i+1] - idx[i]
    return seg_size/seg_count


def windiff_metric_ONE_SEQUENCE(y_true, y_pred, win_size=-1, rounded=True, print_individual_stats=True):
    """ Make sure Y_pred is ROUNDED
    """
    if win_size == -1:
        window_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27,28,29,31]
    else:
        window_sizes = [win_size]

    #print "Window Size:", window_sizes

    #####################################################
    # Remove the padded elements before calculating the
    # windiff metric so that we have better values
    #####################################################

    metric_outputs = []
    assert y_true.shape[0] == y_pred.shape[0]

    print ">>>>> X:", y_true.shape
    print "Avg Seg Length: %f | We use SEG_LEN/2 as the window size" %(compute_avg_seg_len(y_true))
    for window_size in window_sizes:
        ans = -1
        lenn = y_pred.shape[0]
        if not rounded:
            y_pred = round(y_pred)
    
        if window_size <= lenn:
            t_cum = np.cumsum(y_true, axis=0)
            p_cum = np.cumsum(y_pred, axis=0)
            ans_list = []
            for i in range(len(y_true)):
                if i < window_size-1:
                    continue
                elif i == window_size-1:
                    ans_list.append((t_cum[i] - p_cum[i]) != 0)
                else:
                    ans_list.append(((t_cum[i]-t_cum[i-window_size]) - (p_cum[i]-p_cum[i-window_size])) != 0)
            ans = (np.sum(ans_list)*1.0)/(lenn - window_size)
        else:
            print 'ERROR: Window Size larger then total sample length'

        metric_outputs.append({ 'window_size': window_size,
                                'windiff': ans
                                })
    if print_individual_stats:
        windiff_values = [dic['windiff'] for dic in metric_outputs]
        headers = ['****'] + ["Wind=" + str(i) for i in window_sizes]
        print tabulate([["WinDiff values"] + windiff_values], headers=headers)

    return metric_outputs


#def save_model(filename, model):
#    # serialize model to JSON
#    if len(filename.split(".")) > 1:
#        print "Filename '%s' should not contain a '.'" %(filename)
#    model_json = model.to_json()
#    with open(filename + ".json", "w") as json_file:
#        json_file.write(model_json)
#
#    # serialize weights to HDF5
#    filename_save = filename + ".h5"
#    model.save_weights(filename_save)
#    print "Saved model to disk with name '%s'" %(filename_save)
# 
#
#def load_model(filename): 
#    if len(filename.split(".")) > 1:
#        print "Filename '%s' should not contain a '.'" %(filename)
#    # load json and create model
#    with open(filename + ".json", 'r') as json_file:
#        loaded_model_json = json_file.read()
#
#    loaded_model = model_from_json(loaded_model_json)
#    # load weights into new model
#    loaded_model.load_weights(filename + ".h5")
#    print("####### Loaded model from disk!")
#    return loaded_model


#WINDOW_SIZE_windiff_metric = 10
#def window_diff_metric_TENSOR(y_true, y_pred):
#    # y_true.shape = (BATCH_SIZE, INPUT_VECTOR_LENGTH)
#    _padding_var = T.set_subtensor(y_true[0], 0)
#
#    yT = T.concatenate((_padding_var, T.extra_ops.cumsum(y_true, axis=1)))
#    yP = T.concatenate((_padding_var, T.extra_ops.cumsum(y_pred, axis=1)))
#
#    winT = (yT - T.roll(yT, WINDOW_SIZE_windiff_metric))[WINDOW_SIZE_windiff_metric:]
#    winP = (yP - T.roll(yP, WINDOW_SIZE_windiff_metric))[WINDOW_SIZE_windiff_metric:]
#
#    result = T.mean(T.eq(winT - winP, 0))
#    #result = T.sum(T.eq(winT - winP, 0))/y_true.shape[0]
#    return result

