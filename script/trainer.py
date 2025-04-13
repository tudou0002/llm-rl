import os
import d3rlpy
import numpy as np


def get_model(config, algo, dynamics=None):

    if algo == 'COMBO':
        # encoder_factory = CustomVectorEncoderFactory(
        #     config,
        #     action_size=config["action_size"],
        #     mask_size=int(config['page_items'])+1,
        #     with_q=True,
        #     hidden_units=[256]
        # )
        model = d3rlpy.algos.COMBO(
            dynamics = dynamics,
            batch_size=256,
            update_actor_interval = 2000,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # actor_encoder_factory=encoder_factory,
            gamma=1.0,
            scaler = 'min_max',
            reward_scaler='standard',
            # use_gpu=config['gpu'],
            # real_ratio = 1,
            # rollout_interval = 1000000000,
            # generated_maxlen = 100
        )
    elif algo == 'CQL':
        encoder_factory = CustomVectorEncoderFactory(
            config,
            action_size=config["action_size"],
            mask_size=int(config['page_items'])+1,
            with_q=True,
            hidden_units=[256]
        )
        model = d3rlpy.algos.DiscreteCQL(
            batch_size=256,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # encoder_factory=encoder_factory,
            gamma=1.0,
            alpha=config["CQL_alpha"],
            reward_scaler='standard',
            # use_gpu=config['gpu']
        )

    return model

def build_with_dataset(model, dataset):
    model.build_with_dataset(dataset)
    if model._scaler:
        model._scaler.fit(dataset)

        # initialize action scaler
    if model._action_scaler:
        model._action_scaler.fit(dataset)

        # initialize reward scaler
    if model._reward_scaler:
        model._reward_scaler.fit(dataset)
    return model

def d3rlpy_eval(eval_episodes, policy: policy_model, soft_opc_score=90):
    if isinstance(policy.policy, d3rlpy.algos.DiscreteBC):
        scorers = {
            'discrete_action_match': d3rlpy_scorer.discrete_action_match_scorer,
        }
        for name, scorer in scorers.items():
            test_score = scorer(policy, eval_episodes)
            print(name, ' ', test_score)
    else:
        scorers = {
            'discrete_action_match': d3rlpy_scorer.discrete_action_match_scorer,
            'soft_opc': d3rlpy_scorer.soft_opc_scorer(soft_opc_score)
        }
        for name, scorer in scorers.items():
            test_score = scorer(policy, eval_episodes)
            print(name, ' ', test_score)


def evaluate(config, policy: policy_model):
    eval_config = config.copy()
    eval_config["is_eval"] = True
    eval_config["batch_size"] = 2048
    max_steps = eval_config["max_steps"]
    if eval_config['env'] == 'SlateRecEnv-v0':
        sim = SlateRecEnv(eval_config, state_cls=SlateState)
        eval_env = gym.make('SlateRecEnv-v0', recsim=sim)
    elif eval_config['env'] == 'SeqSlateRecEnv-v0':
        sim = SeqSlateRecEnv(eval_config, state_cls=SeqSlateState)
        eval_env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    else:
        assert eval_config['env'] in ('SlateRecEnv-v0', 'SeqSlateRecEnv-v0')

    epoch = 4
    episode_rewards, prev_actions = [], []
    for i in range(epoch):
        obs = eval_env.reset()
        episode_reward = []
        print('test batch at ', i)
        for j in range(max_steps):
            action = policy.predict_with_mask(obs)
            obs, reward, done, info = eval_env.step(action)
            episode_reward.append(reward)
            prev_actions.append(action)

        episode_reward = np.sum(np.array(episode_reward), axis=0)
        episode_rewards.append(episode_reward)

    episode_rewards = np.array(episode_rewards)
    print('reward max min', np.max(episode_rewards), np.min(episode_rewards))
    print('avg reward', np.sum(episode_rewards) / eval_config['batch_size'] / epoch)