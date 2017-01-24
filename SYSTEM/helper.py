import numpy as np
import theano.tensor as T



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


def windiff_metric_NUMPY(y_true, y_pred, window_size=10):

    """ Make sure Y_pred is ROUNDED
    """
    WINDOW_SIZE_windiff_metric = window_size
    print "Window Size:", window_size

    #####################################################
    # Remove the padded elements before calculating the
    # windiff metric so that we have better values
    #####################################################
    ans = []
    window = WINDOW_SIZE_windiff_metric
    for sample_T, sample_P in zip(y_true, y_pred):
        #print sample_T.shape, sample_P.shape
        if WINDOW_SIZE_windiff_metric > sample_T.shape[0]:
            print 'ERROR: Window Size larger then total sample length'
            return -1

        t_cum = np.cumsum(sample_T, axis=0)
        p_cum = np.cumsum(sample_P, axis=0)
        ans_list = []
        for i in range(len(sample_T)):
            if i < window-1:
                continue
            elif i == window-1:
                ans_list.append((t_cum[i] - p_cum[i]) != 0)
            else:
                ans_list.append(((t_cum[i]-t_cum[i-window]) - (p_cum[i]-p_cum[i-window])) != 0)
        ans.append((np.sum(ans_list)*1.0)/(len(sample_T) - window))

    return {
        'mean': np.mean(ans),
        'std': np.std(ans)
        }


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

