Q1

Ant

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--eval_batch_size 4000 \
--n_layers 3 \
--hidden 64 \
--learning_rate 0.01

Hopper

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name bc_hopper --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 \
--eval_batch_size 4000 \
--n_layers 3 \
--hidden 64 \
--learning_rate 0.01

Q2

Ant

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--eval_batch_size 4000 \
--n_layers 3 \
--hidden 64 \
--learning_rate 0.01

Hopper

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name dagger_hopper --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 \
--eval_batch_size 4000 \
--n_layers 3 \
--hidden 64 \
--learning_rate 0.01


