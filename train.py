from run_maml import *

run_maml(n_way=5, k_shot=3, inner_update_lr=0.4, num_inner_updates=1, meta_train_iterations=5000, logdir='./log/model')
# run_maml(n_way=5, k_shot=1, inner_update_lr=0.4, num_inner_updates=1, meta_train=False, logdir='./log/model/')