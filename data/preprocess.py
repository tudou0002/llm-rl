from d3rlpy.dataset import MDPDataset
import d3rlpy
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder 
from openai import OpenAI
import random
import tiktoken
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
client = OpenAI()

# the in-/out-/request price in USD per M-token
# for ours
MODEL_PRICE = {'gpt-4o-mini':   [0.15, 0.6, 0],
                'gpt-4o':        [5, 15, 0],
                'gemini-1.5-flash-002':  [0.075, 0.3, 0],
                'gemini-1.5-pro-002':    [3.5, 10.5, 0],
                'gemini-1.0-pro':    [0.5, 1.5, 0],
                'Phi-3-mini-4k-instruct':     [0.13, 0.52, 0],
                'Phi-3.5-mini-instruct':   [0.13, 0.52, 0],
                'Phi-3-small-8k-instruct':    [0.15, 0.6, 0],
                'Phi-3-medium-4k-instruct':  [0.17, 0.68, 0],
                'llama-3-8B':     [0.055, 0.055, 0],
                'llama-3-70B':    [0.35, 0.4, 0],
                'mixtral-8x7B':  [0.24, 0.24, 0]
                }
ACTION_IDX = {'gpt-4o-mini': 0,
                'gpt-4o':  1,
                'gemini-1.5-flash-002': 2,
                'gemini-1.5-pro-002':   3,
                'gemini-1.0-pro':   4,
                'Phi-3-mini-4k-instruct':    5,
                'Phi-3.5-mini-instruct':  6,
                'Phi-3-small-8k-instruct':   7,
                'Phi-3-medium-4k-instruct': 8,
                'llama-3-8B':    9,
                'llama-3-70B':   10,
                'mixtral-8x7B': 11,
                'terminate': 12
              }
IDX_TO_ACTION = {v: k for k, v in ACTION_IDX.items()}

def train_vectorizer(dataset_name):
    df = load_dataset(dataset_name, 'train')
    # train a one hot encoder on answers and actions
    # we need to convert the answers and actions to a list of strings
    answers = df['ref_answer'].tolist()
    answers.append(np.nan)
    actions = ACTIONS_SPACE
    actions = np.array(actions).reshape(-1,1)
    answers = np.array(answers).reshape(-1,1)

    # train a one hot encoder on the answers and actions
    ohe_actions = OneHotEncoder()
    ohe_actions.fit(actions)
    ohe_answers = OneHotEncoder()
    ohe_answers.fit(answers)
    return ohe_actions, ohe_answers

def vectorize_history(history,ohe_actions, ohe_answers, max_iter=5,  method='ohe'):
    # history is a string of the form "model1: answer1; model2: answer2; ..."
    # we want to vectorize it into a numpy array 
    # we use the one hot encoding to vectorize the history
    if method == 'ohe':
        # split the history string into a list of model:answer pairs
        if history == '':
            return np.zeros((len(ohe_actions.categories_[0]) + len(ohe_answers.categories_[0]))*max_iter)  
        pairs = history.split('; ')
        # print(history)
        # create a dictionary to store the one hot encoded history
        history_dict = {}
        history_vec = []
        for pair in pairs:
            model, answer = pair.split(': ')
            history_dict[model] = answer
            action_vec = ohe_actions.transform(np.array([model]).reshape(1,-1)).toarray()
            answer_vec = ohe_answers.transform(np.array([answer]).reshape(1,-1)).toarray()
            # print('action_vec', action_vec[0], 'answer_vec', answer_vec[0])
            history_vec.extend(action_vec[0])
            history_vec.extend(answer_vec[0])
        # print('history_vec', history_vec, 'history_vec shape', len(history_vec))
        # fill up the remaining slots with zeros until max_iter
        try:
            history_vec = np.concatenate([history_vec, np.zeros(max_iter*(len(ohe_actions.categories_[0]) + len(ohe_answers.categories_[0])) - len(history_vec))])
        except:
            print('history_vec', history_vec, 'history_vec shape', len(history_vec))
            print('max_iter', max_iter)
            print('ohe_actions.categories_[0]', len(ohe_actions.categories_[0]))
            print('ohe_answers.categories_[0]', len(ohe_answers.categories_[0]))
    # print('history_vec shape', history_vec.shape)
    return history_vec

def price_of(q, model):
    # we assume the output size is constantly 1 for classification problems
    assert model in MODEL_PRICE, f'wrong: {model} not supported'
    # we simply apply the tiktoken to count tht token number for all models
    encoding = tiktoken.get_encoding('cl100k_base')
    in_token_num = len(encoding.encode(q))
    # for classification tasks, we assume the model always give out answer with len=1
    out_token_num = 1
    in_price = MODEL_PRICE[model][0] * in_token_num / 1e6
    out_price = MODEL_PRICE[model][1] * out_token_num / 1e6
    request_price = MODEL_PRICE[model][2]
    return in_price + out_price + request_price

def price_all(q, models):
    costs = []
    for m in models:
        costs.append(price_of(q, m))
    return costs

def load_dataset(dataset_name, split):

    df = pd.read_csv(f'{dataset_name}/embedded_{dataset_name}_all_models_clean_{split}.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    global ACTIONS_SPACE
    ACTIONS_SPACE = list(set(df.columns.tolist()) - set(['query','query_raw', 'ref_answer', 'embedding']))+['terminate']
    # convert all action space columns to string 
    
    if dataset_name == 'SCIQ':
        for col in list(set(df.columns.tolist()) - set(['query','query_raw', 'embedding'])):
            df[col] = df[col].astype('Int64').astype(str).replace('<NA>', np.nan)

    global ORDERED_ACTIONS_SPACE
    ORDERED_ACTIONS_SPACE = ACTIONS_SPACE
    ORDERED_ACTIONS_SPACE.remove('terminate')
    ORDERED_ACTIONS_SPACE.sort(key=lambda x: MODEL_PRICE[x][0])
    ORDERED_ACTIONS_SPACE.append('terminate')
    return df

# actions = [ 'gpt-4o-mini', 'gpt-4o',
#        'llama-3-8B', 'llama-3-70B', 'mixtral-8x7B', 'gemini-1.5-flash-002',
#        'gemini-1.0-pro', 'gemini-1.5-pro-002', 'Phi-3.5-mini-instruct',
#        'Phi-3-small-8k-instruct', 'Phi-3-mini-4k-instruct',
#        'Phi-3-medium-4k-instruct', 'terminate']

def compute_aggregate_reward(history, ref_answer, query):
    # Count votes for each answer
    vote_counts = {}
    model_costs = {}
    
    for model, answer in history.items():
        # Strip any whitespace and normalize the answer
        try:
            normalized_answer = answer.strip()
        except:
            normalized_answer = 'nan'
        
        # Add to vote count
        if normalized_answer not in vote_counts:
            vote_counts[normalized_answer] = 1
            model_costs[normalized_answer] = price_of(query, model)
        else:
            vote_counts[normalized_answer] += 1
            model_costs[normalized_answer] += price_of(query, model)
        
    
    # Find the answer(s) with the most votes
    max_votes = max(vote_counts.values())
    top_answers = [ans for ans, votes in vote_counts.items() if votes == max_votes]
    
    # If there's a tie, choose the answer from the highest-cost model
    if len(top_answers) > 1:
        final_answer = max(top_answers, key=lambda ans: model_costs[ans])
    else:
        final_answer = top_answers[0]
    return 1 if final_answer == ref_answer else -1

# Function to create a single episode
def create_episode(row, max_budget, max_iter=5, ohe_actions=None, ohe_answers=None, strategy='random'):
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    query = row['query_raw']
    ref_answer = row['ref_answer']
    
    # Initial state: query embedding + empty history + remaining budget
    history = {}
    remaining_budget = max_budget
    
    # Get query embedding by using the embedding column in the dataframe
    query_embedding = row['embedding']
    queried_actions = set()
    
    # Create initial observation (query embedding + history + remaining budget)
    # For history, we'll flatten it into a string representation
    history_str = ""
    current_obs = np.concatenate([
        query_embedding, # shape: (512,)
        vectorize_history(history_str, ohe_actions, ohe_answers, method='ohe' ),
        np.array([remaining_budget])
    ])
    
    # Episode continues until termination or budget exhausted or max_iter reached
    terminated = 0
    iter = 0
    while not terminated and remaining_budget > 0 and len(actions) < max_iter:
        observations.append(current_obs)
        
        # Choose a random action from the action space
        if strategy == 'random':
            available_actions = set(ACTIONS_SPACE)- queried_actions 
            selected_action = random.choice(list(available_actions))
            # if first action is terminate, we need to resample the action
            if selected_action == 'terminate' and (not history):
                continue
        elif strategy == 'cascade':
            available_actions = set(ACTIONS_SPACE)- queried_actions - {'terminate'}
            if available_actions == set():
                selected_action = 'terminate'
            elif available_actions != set() and iter <=max_iter:
                # select the cheapest action 
                selected_action = min(available_actions, key=lambda x: MODEL_PRICE[x][0])
                # if the cheapest action is too expensive, set to be terminate
                if price_of(query, selected_action) > remaining_budget:
                    selected_action = 'terminate'
            elif available_actions != set() and iter > max_iter:
                selected_action = 'terminate'
        elif strategy == 'greedy':
            # choose the action with the highest cost under the remaining budget
            # we need to compute the cost of all actions
            available_actions = set(ACTIONS_SPACE)- queried_actions - {'terminate'}
            # print('available_actions', available_actions)
            if available_actions == set():
                selected_action = 'terminate'
            elif available_actions != set() and iter <= max_iter:
                selected_action = max(available_actions, key=lambda x: MODEL_PRICE[x][0])
                # select the most expensive action if it is under the remaining budget
                while price_of(query, selected_action) > remaining_budget:
                    available_actions.remove(selected_action)
                    if available_actions == set():
                        # even the most expensive action is too expensive
                        selected_action = 'terminate'
                        break
                    selected_action = max(available_actions, key=lambda x: MODEL_PRICE[x][0])
                if available_actions == set():
                    # even the most expensive action is too expensive
                    selected_action = 'terminate'
            elif available_actions != set() and iter > max_iter:
                selected_action = 'terminate'
            

        queried_actions.add(selected_action)
        actions.append(ACTION_IDX[selected_action])
        iter += 1
        if selected_action == 'terminate':
            # set the termination flag if the action is terminate
            terminated = 1

        if strategy == 'cascade':
            # if current history can be aggregated to the correct answer, we terminate
            if terminated != 1 and history :
                if compute_aggregate_reward(history, ref_answer, query) == 1:
                    terminated = 1
                    selected_action = 'terminate'
        
        if terminated == 1:
            # Check if the final answer matches reference
            # We'll use the last model's answer as our final answer if history is not empty
            # Ensemble voting for classification answers
            if history:
                terminal_reward = compute_aggregate_reward(history, ref_answer, query)
                terminal_reward = terminal_reward + sum(rewards)
                
                rewards.append(terminal_reward)
            else:
                # No models were queried before termination
                terminal_reward = -1 
                # if -sum(rewards)/10000 > max_budget:
                # terminal_reward = terminal_reward + sum(rewards) 
                rewards.append(terminal_reward)
            terminals.append(1)
            return observations, actions, rewards, terminals
        else:
            # Model query action
            if selected_action in ACTIONS_SPACE:
                model_answer = row[selected_action]
                
                query_cost = price_of(query, selected_action)
                remaining_budget -= query_cost
                
                history[selected_action] = model_answer
                
                history_str = "; ".join([f"{model}: {answer}" for model, answer in history.items()])
                
                rewards.append(-query_cost*10000)  # scale up the cost
                
                current_obs = np.concatenate([
                    query_embedding,
                    vectorize_history(history_str, ohe_actions, ohe_answers, method='ohe'),
                    np.array([remaining_budget])
                ])
            else:
                # Model not in dataset, penalize
                rewards.append(-1)  # Small penalty for invalid action
        
        # Set terminal flag
        terminals.append(terminated)
    # append the termination observation and action
    terminals.append(1)
    actions.append(len(ACTIONS_SPACE)-1)
    terminal_reward = compute_aggregate_reward(history, ref_answer, query)
    # if -sum(rewards)/10000 > max_budget:
    terminal_reward = terminal_reward + sum(rewards) 
    # else:
    #     terminal_reward = terminal_reward
    rewards.append(terminal_reward)
    observations.append(current_obs)

    return observations, actions, rewards, terminals

# Function to create a single episode
def evaluate_row(row, max_budget, model, max_iter=5, ohe_actions=None, ohe_answers=None):
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    query = row['query_raw']
    ref_answer = row['ref_answer']
    
    # Initial state: query embedding + empty history + remaining budget
    history = {}
    remaining_budget = max_budget
    
    # Get query embedding by using the embedding column in the dataframe
    query_embedding = row['embedding']
    
    # Create initial observation (query embedding + history + remaining budget)
    # For history, we'll flatten it into a string representation
    history_str = ""
    current_obs = np.concatenate([
        query_embedding, # shape: (512,)
        vectorize_history(history_str, ohe_actions, ohe_answers, method='ohe' ),
        np.array([remaining_budget])
    ])
    queried_actions = set()
    
    # Episode continues until termination or budget exhausted or max_iter reached
    terminated = 0
    iter = 0

    while not terminated and remaining_budget > 0 and iter < max_iter:
        observations.append(current_obs)

        action_idx = model.predict(current_obs.reshape(1, -1))[0]
        # print('action_idx', action_idx)

        selected_action = IDX_TO_ACTION[action_idx]
        iter += 1
        if selected_action in queried_actions:
            # if the action is already queried, we need to return 
            terminal_reward = compute_aggregate_reward(history, ref_answer, query)
            if terminal_reward == 1:
                return ref_answer 
            else:
                return ''
        queried_actions.add(selected_action)

        # if first action is terminate, we need to return empty as the predicted result
        if history == {} and selected_action == 'terminate':
            return ''
        actions.append(action_idx)
        
        
        if selected_action == 'terminate':
            # Check if the final answer matches reference
            # We'll use the last model's answer as our final answer if history is not empty
            # Ensemble voting for classification answers
            terminated = 1
            if history:
                terminal_reward = compute_aggregate_reward(history, ref_answer, query)
                # rewards.append(terminal_reward)
                if terminal_reward == 1:
                    return ref_answer 
                else:
                    return ''
            else:
                # No models were queried before termination
                # rewards.append(-10)
                return ''
            # terminals.append(1)
            # return observations, actions, rewards, terminals
        else:
            # Model query action
            if selected_action in ACTIONS_SPACE:
                model_answer = row[selected_action]
                
                query_cost = price_of(query, selected_action)
                remaining_budget -= query_cost
                
                history[selected_action] = model_answer
                
                history_str = "; ".join([f"{model}: {answer}" for model, answer in history.items()])
                
                rewards.append(-query_cost*10000)  # scale up the cost
                
                current_obs = np.concatenate([
                    query_embedding,
                    vectorize_history(history_str, ohe_actions, ohe_answers, method='ohe'),
                    np.array([remaining_budget])
                ])
            else:
                # Model not in dataset, penalize
                rewards.append(-1)  # Small penalty for invalid action
        
        # Set terminal flag
        terminals.append(terminated)
    # append the termination observation and action
    terminals.append(1)
    actions.append(len(ACTIONS_SPACE)-1)
    rewards.append(compute_aggregate_reward(history, ref_answer, query))
    observations.append(current_obs)

    # return observations, actions, rewards, terminals
    return ref_answer if rewards[-1] == 1 else ''

def evaluate(dataset_name, split, model, budget, ohe_actions, ohe_answers):
    df = load_dataset(dataset_name, split)
    correct = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        ref_answer = row['ref_answer']
        predicted_answer = evaluate_row(row, budget,model, 5,  ohe_actions, ohe_answers)
        if predicted_answer == ref_answer:
            correct += 1
    print(f'accuracy for RL: {correct/len(df)}')
    for model in MODEL_PRICE.keys():
        print(f'model accuracy for {model}:', (df[model] == df['ref_answer']).mean())
    

def create_dataset(dataset_name, split, mode,ohe_actions, ohe_answers, max_iter=5):
    # Generate episodes
    episodes = []
    observations = []
    actions_idxs = []
    rewards = []
    terminals = []
    df = load_dataset(dataset_name, split)
    # use tqdm to iterate over the dataframe
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # random.seed(i)
        # Randomly set a budget between 0.0005 and 0.005
        max_budget = random.uniform(0.00001, 0.001)
        
        observation, actions_idx, reward, terminal = create_episode(row, max_budget, max_iter=max_iter, 
                                                                    ohe_actions=ohe_actions, ohe_answers=ohe_answers, 
                                                                    strategy='cascade')
        if len(observation) > 0:  # Only add non-empty episodes
            observations.extend(np.array(observation))
            actions_idxs.extend(np.array(actions_idx))
            rewards.extend(np.array(reward))
            terminals.extend(np.array(terminal))

        if 'R' in mode:
            observation, actions_idx, reward, terminal = create_episode(row, max_budget, max_iter=max_iter, 
                                                                    ohe_actions=ohe_actions, ohe_answers=ohe_answers, 
                                                                    strategy='random')
            if len(observation) > 0:  # Only add non-empty episodes
                observations.extend(np.array(observation))
                actions_idxs.extend(np.array(actions_idx))
                rewards.extend(np.array(reward))
                terminals.extend(np.array(terminal))

        if 'G' in mode:
            observation, actions_idx, reward, terminal = create_episode(row, max_budget, max_iter=max_iter, 
                                                                    ohe_actions=ohe_actions, ohe_answers=ohe_answers, 
                                                                    strategy='greedy')
            if len(observation) > 0:  # Only add non-empty episodes
                observations.extend(np.array(observation))
                actions_idxs.extend(np.array(actions_idx))
                rewards.extend(np.array(reward))
                terminals.extend(np.array(terminal))
                
    print('observations', np.stack(observations, axis=0).shape)
    print('actions_idxs', np.array(actions_idxs).shape)
    print('rewards', np.array(rewards).shape)
    print('terminals', np.array(terminals).shape)

    # Create MDPDataset
    mdp_dataset = MDPDataset(observations=np.stack(observations, axis=0, dtype=np.float32), actions=np.array(actions_idxs), 
                            rewards=np.array(rewards).astype(np.float32), terminals=np.array(terminals).astype(np.float32))

    # Save the dataset
    mdp_dataset.dump(f'{dataset_name}/{dataset_name}_rl_dataset_{split}_{mode}.h5')
    print(f"Created offline RL dataset with {len(episodes)} episodes")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='OVERRULING')
    args.add_argument('--split', type=str, default='train')
    args.add_argument('--model_path', type=str, default='')
    args.add_argument('--evaluate', type=str, default='')
    args.add_argument('--max_iter', type=int, default=5)
    args.add_argument('--budget', type=float, default=1e-5)
    args.add_argument('--mode', type=str, default='C', choices=['C', 'CR', 'CRG'])
    args = args.parse_args()

    ohe_actions, ohe_answers = train_vectorizer(args.dataset)
    if not args.evaluate:
        create_dataset(args.dataset, args.split, args.mode, ohe_actions, ohe_answers, args.max_iter)
    elif args.evaluate=='True':
        model = d3rlpy.load_learnable(args.model_path)
        evaluate(args.dataset, args.split, model, args.budget, ohe_actions, ohe_answers)

    # python preprocess.py --dataset OVERRULING --split test --model_path ../script/d3rlpy_logs/cql_OVERRULING_CR_20250412161340/model_20000.d3 --evaluate True --budget 1e-6
    # python preprocess.py --dataset OVERRULING --split train --mode C
    # python preprocess.py --dataset HEADLINES --split train --mode C
    # python preprocess.py --dataset AGNEWS --split train --mode C
    # python preprocess.py --dataset SCIQ --split train --strategy cascade

