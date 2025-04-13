import os
# import gym
import random
import d3rlpy
import sys
import torch
# from rl4rs.env.slate import SlateRecEnv, SlateState
# from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState
from script import batchrl_trainer
from d3rlpy.dataset import MDPDataset
from script.offline_evaluation import ope_eval
# from rl4rs.policy.behavior_model import behavior_model
# from rl4rs.policy.policy_model import policy_model
# from rl4rs.nets.cql.encoder import CustomVectorEncoderFactory
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer

algo = sys.argv[1]
stage = sys.argv[2]
extra_config = eval(sys.argv[3]) if len(sys.argv) >= 4 else {}

config = {"epoch": 4, "maxlen": 64, "batch_size": 2048, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128,
          "hidden_units": 128, "max_steps": 9, "sample_file": '../dataset/rl4rs_dataset_a_shuf.csv',
          "model_file": "../output/rl4rs_dataset_a_dnn/model", 'gpu': True, "page_items": 9, 'action_emb_size':32,
          "iteminfo_file": '../dataset/item_info.csv', "support_d3rl_mask": True, "is_eval": True,
          "CQL_alpha": 1, 'trial_name': 'a_all'}

config = dict(config, **extra_config)


if algo in ('MOPO', 'COMBO') or 'conti' in algo:
    config["support_conti_env"] = True

if not config.get('gpu', True):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    torch.cuda.is_available = lambda: False
    print('CUDA_VISIBLE_DEVICES', torch.cuda.is_available())

if not config.get("support_conti_env",False):
    trail_name = config['env'] + '_' + config['trial_name'] + '.h5'
elif config.get("support_onehot_action", False):
    config['action_emb_size'] = config["action_size"]
    trail_name = config['env'] + '_' + config['trial_name'] + '_onehot.h5'
else:
    trail_name = config['env'] + '_' + config['trial_name'] + '_conti.h5'
dataset_dir = os.environ['rl4rs_dataset_dir']
output_dir = os.environ['rl4rs_output_dir']
dataset_save_path = dataset_dir + '/' + trail_name
dynamics_save_path = output_dir + '/' + 'dynamics' + '_' + trail_name
model_save_path = output_dir + '/' + algo + '_' + trail_name
scaler = None
print(trail_name, config)


def load_dataset(args.datafile):
    with open(args.datafile, "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    return dataset

try:
    dynamics = batchrl_trainer.get_model(config, 'dynamics')
    dynamics = batchrl_trainer.build_with_dataset(dynamics, dataset)
    dynamics.load_model(dynamics_save_path)
except Exception:
    dynamics = None

def main(args):
    if args.stage == 'train':
        model = batchrl_trainer.get_model(config, algo, dynamics)
        model.fit(dataset,
                eval_episodes=dataset.episodes[-3000:],
                n_epochs=config['epoch'],
                show_progress=False)
        model.save_model(model_save_path)

    if args.stage == 'eval':
        default_soft_opc_score = 90 \
            if config['env'] == 'SlateRecEnv-v0' \
            else 90 * 2
        soft_opc_score = config.get('soft_opc_score', default_soft_opc_score)
        model = batchrl_trainer.get_model(config, algo, dynamics)
        model = batchrl_trainer.build_with_dataset(model, dataset)
        model.load_model(model_save_path)
        eval_episodes = random.sample(dataset.episodes, 2048 * 4)
        policy = policy_model(model, config=config)
        # batchrl_trainer.d3rlpy_eval(eval_episodes, policy, soft_opc_score)
        batchrl_trainer.evaluate(config, policy)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--algo', type=str, default='MOPO')
    args.add_argument('--stage', type=str, default='train')
    args.add_argument('--datafile', type=str, default='HEADLINES')
    # args.add_argument('--extra_config', type=str, default='{}')
    args = args.parse_args()
    main(args)