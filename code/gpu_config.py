import os
import subprocess
import pdb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

command = ['nvidia-smi', '--format=csv', '--query-gpu=memory.free']
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()

out = [line for line in out.split('\n')[1:] if len(line.strip())]
out = [(int(line.split()[0]), i) for i, line in enumerate(out)]
out = sorted(out, key=lambda x:x[0], reverse=True)

select_gpu = out[0][1]
os.environ["CUDA_VISIBLE_DEVICES"]=str(select_gpu)


# Set memory limit
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


print "#"*40
print "   Using GPU %d for computation  "%(select_gpu)
print "   allow_memory_groth=True to avoid full memory allocation    "
print "#"*40

