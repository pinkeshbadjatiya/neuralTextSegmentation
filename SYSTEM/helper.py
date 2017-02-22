import numpy as np
import theano.tensor as T
import pdb


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

def windiff_metric_NUMPY(y_true, y_pred, win_size=10, rounded=True):
    """ Make sure Y_pred is ROUNDED
    """
    if win_size == -1:
        window_sizes = [9,11,13,15,17,19,21,23,25,27,28,29,31]
    else:
        window_sizes = [win_size]

    print "Window Size:", window_sizes

    #####################################################
    # Remove the padded elements before calculating the
    # windiff metric so that we have better values
    #####################################################

    metric_outputs = []
    for window_size in window_sizes:
        ans = []
        for sample_T, sample_P in zip(y_true, y_pred):
            sample_P = np.array(sample_P)
            sample_P = sample_P.reshape((sample_P.shape[0], 1))      # Convert from (SAMPLE, 37, 1, 1) -> (SAMPLE, 37, 1)
            if not rounded:
                sample_P = round(sample_P)
    
            #print sample_T.shape, sample_P.shape
            if window_size > sample_T.shape[0]:
                print 'ERROR: Window Size larger then total sample length'
                return -1

            t_cum = np.cumsum(sample_T, axis=0)
            p_cum = np.cumsum(sample_P, axis=0)
            ans_list = []
            for i in range(len(sample_T)):
                if i < window_size-1:
                    continue
                elif i == window_size-1:
                    ans_list.append((t_cum[i] - p_cum[i]) != 0)
                else:
                    ans_list.append(((t_cum[i]-t_cum[i-window_size]) - (p_cum[i]-p_cum[i-window_size])) != 0)
            ans.append((np.sum(ans_list)*1.0)/(len(sample_T) - window_size))

        metric_outputs.append({ 'window_size': window_size,
                                'mean': np.mean(ans),
                                'std': np.std(ans)
                                })
    return metric_outputs


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

