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
    # Assuming segment_length is the no of sentences in a section/segment
    idx = np.where(y_true == 1)[0]
    seg_sizes, seg_count = [], idx.shape[0]
    for i in range(seg_count):
        if i == seg_count - 1:
            seg_sizes.append(y_true.shape[0] - idx[i])
        else:
            seg_sizes.append(idx[i+1] - idx[i])
    return np.mean(seg_sizes)


def windiff_and_pk_metric_ONE_SEQUENCE(y_true, y_pred, window_size=-1, rounded=True, print_individual_stats=True):
    """ Make sure Y_pred is ROUNDED
    """
    #####################################################
    # Remove the padded elements before calculating the
    # windiff metric so that we have better values
    #####################################################

    # NOTE: WindowSize is the no of boundaries between the ith and jth sentence. i.e. no of sentences in between + 1

    assert y_true.shape[0] == y_pred.shape[0]

    average_seg_length = compute_avg_seg_len(y_true)
    if window_size == -1:
        window_size = int(average_seg_length * 0.5)   # WindowSize is equal to 1/2 of average window size of that document

    lenn = y_pred.shape[0]
    if not rounded:
        y_pred = round(y_pred)

    t_cum, p_cum = np.cumsum(y_true, axis=0), np.cumsum(y_pred, axis=0)
    ans_list = {
          'wd': [],
          'pk': []
        }
    for i in range(0, lenn - window_size):
        ans_list['wd'].append((sum(y_true[i+1: i+window_size+1]) - sum(y_pred[i+1: i+window_size+1])) != 0)   #### IMP. IMP. IMP.
        ans_list['pk'].append(((t_cum[i]-t_cum[i+window_size]) - (p_cum[i]-p_cum[i+window_size])) != 0)

    ans = {}
    ans['wd'] = (np.sum(ans_list['wd'])*1.0)/(lenn - window_size)
    ans['pk'] = (np.sum(ans_list['pk'])*1.0)/(lenn - window_size)

    if print_individual_stats:
        print ">> X:", y_true.shape, "| Avg_Seg_Length: %f | WinDiff: %f | Pk: %f" %(average_seg_length, ans['wd'], ans['pk'])

    return average_seg_length, ans


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

