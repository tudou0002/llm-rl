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
from preprocess import *
import matplotlib.pyplot as plt
import seaborn as sns

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

ACTIONS_SPACE = ACTION_IDX.keys()


def evaluate_row(row, max_budget, model, max_iter=5, ohe_actions=None, ohe_answers=None):
    observations = []
    actions = []
    rewards = []
    terminals = []

    spent_cost = 0
    
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
                return ref_answer, spent_cost
            else:
                return '', spent_cost
        queried_actions.add(selected_action)

        # if first action is terminate, we need to return empty as the predicted result
        if history == {} and selected_action == 'terminate':
            return '', spent_cost
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
                    return ref_answer, spent_cost
                else:
                    return '', spent_cost
            else:
                # No models were queried before termination
                # rewards.append(-10)
                return '', spent_cost
            # terminals.append(1)
            # return observations, actions, rewards, terminals
        else:
            # Model query action
            if selected_action in ACTIONS_SPACE:
                model_answer = row[selected_action]
                
                query_cost = price_of(query, selected_action)
                spent_cost += query_cost
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
    return ref_answer if rewards[-1] == 1 else '', spent_cost

def evaluate(dataset_name, split, model, ohe_actions, ohe_answers):
    df = load_dataset(dataset_name, split)
    rl_results = {}
    baseline_results = {}
    overall_spent_costs = {}
    for c_budget in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        correct = 0
        spent_costs = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            ref_answer = row['ref_answer']
            predicted_answer, spent_cost = evaluate_row(row, c_budget,model, 5,  ohe_actions, ohe_answers)
            if predicted_answer == ref_answer:
                correct += 1
            spent_costs.append(spent_cost)
        rl_results[c_budget] = correct/len(df)
        print(f'AVG spent cost for RL when budget is {c_budget}: {sum(spent_costs)/len(spent_costs)}')
        print(f'accuracy for RL when budget is {c_budget}: {correct/len(df)}')
        overall_spent_costs[c_budget] = sum(spent_costs)/len(spent_costs)
    for model in MODEL_PRICE.keys():
        baseline_results[model] = (df[model] == df['ref_answer']).mean()
        print(f'model accuracy for {model}:', (df[model] == df['ref_answer']).mean())
    return rl_results, baseline_results, overall_spent_costs

def plot_results(rl_res, baseline_res, dataset_name, split, traj_type):
    budgets = list(rl_res.keys())
    rl_results = list(rl_res.values())

    # Create a DataFrame for RL results
    rl_df = pd.DataFrame({
        'Budget': budgets,
        'Accuracy': rl_results,
        'Model': 'RL'
    })

    # Create a DataFrame for baseline results
    baseline_dfs = []
    for model, accuracy in baseline_res.items():
        baseline_df = pd.DataFrame({
            'Budget': budgets,
            'Accuracy': [accuracy] * len(budgets),
            'Model': model
        })
        baseline_dfs.append(baseline_df)

    baseline_df = pd.concat(baseline_dfs)

    results_df = pd.concat([rl_df, baseline_df])
    results_df.to_csv(f'../results/{dataset_name}_{split}_{traj_type}_results.csv', index=False)
    baseline_df.to_csv(f'../results/{dataset_name}_{split}_{traj_type}_baseline_results.csv', index=False)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='Budget', y='Accuracy', hue='Model', marker='o', linestyle='--')
    sns.lineplot(data=rl_df, x='Budget', y='Accuracy', marker='o', linestyle='-')
    plt.xscale('log')
    plt.title(f'{dataset_name} {split}')
    plt.xlabel('Cost Budget')
    plt.ylabel('Accuracy')
    plt.legend(title='Model', loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.savefig(f'../plots/{dataset_name}_{split}_{traj_type}_plot.png')

    return   

def plot_spent_cost(overall_spent_costs, dataset_name, split, traj_type):
    budgets = list(overall_spent_costs.keys())
    spent_costs = list(overall_spent_costs.values())
    # draw a diagonal line where x=y 
    plt.figure(figsize=(6, 5))
    plt.plot(budgets, budgets, '--', alpha=0.8, label='Cost Budget Threshold')
    plt.scatter(budgets, spent_costs, color='red', zorder=5, label='Average Spent Cost')
    plt.plot(budgets, spent_costs, marker='o', linestyle='-')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{dataset_name} {split} Cost Budget vs Average Spent Cost')
    plt.xticks(budgets, labels=[f"$10^{{{int(np.log10(v))}}}$" for v in budgets])
    plt.xlabel('Cost Budget')
    plt.legend()
    plt.ylabel('Average Spent Cost')
    plt.xlim(0, 1e-1+1e-1)
    plt.ylim(0, 1e-1+1e-1)
    plt.savefig(f'../plots/{dataset_name}_{split}_{traj_type}_spent_cost_plot.png')
    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='OVERRULING')
    args.add_argument('--split', type=str, default='train')
    args.add_argument('--model_path', type=str, default='')
    args.add_argument('--max_iter', type=int, default=5)
    args.add_argument('--traj_type', type=str)
    args = args.parse_args()

    ohe_actions, ohe_answers = train_vectorizer(args.dataset)
    model = d3rlpy.load_learnable(args.model_path)
    rl_results, baseline_results, overall_spent_costs = evaluate(args.dataset, args.split, model, ohe_actions, ohe_answers)
    plot_results(rl_results, baseline_results, args.dataset, args.split, args.traj_type)
    plot_spent_cost(overall_spent_costs, args.dataset, args.split, args.traj_type)

    exit()

# python evaluate.py --dataset OVERRULING --split test --model_path ../script/d3rlpy_logs/cql_OVERRULING_C/model_20000.d3 --traj_type C
# python evaluate.py --dataset SCIQ --split test --model_path ../script/d3rlpy_logs/cql_SCIQ_C/model_20000.d3 --traj_type C 
# python evaluate.py --dataset SCIQ --split test --model_path ../script/d3rlpy_logs/cql_SCIQ_CR/model_50000.d3 --traj_type CR
# python evaluate.py --dataset SCIQ --split test --model_path ../script/d3rlpy_logs/cql_SCIQ_CRG/model_80000.d3 --traj_type CRG
# python evaluate.py --dataset HEADLINES --split test --model_path ../script/d3rlpy_logs/cql_HEADLINES_C/model_20000.d3 --traj_type C
# python evaluate.py --dataset HEADLINES --split test --model_path ../script/d3rlpy_logs/cql_HEADLINES_CR/model_50000.d3 --traj_type CR
# python evaluate.py --dataset HEADLINES --split test --model_path ../script/d3rlpy_logs/cql_HEADLINES_CRG/model_80000.d3 --traj_type CRG
# python evaluate.py --dataset AGNEWS --split test --model_path ../script/d3rlpy_logs/cql_AGNEWS_C/model_20000.d3 --traj_type C
# python evaluate.py --dataset AGNEWS --split test --model_path ../script/d3rlpy_logs/cql_AGNEWS_CR/model_50000.d3 --traj_type CR
# python evaluate.py --dataset AGNEWS --split test --model_path ../script/d3rlpy_logs/cql_AGNEWS_CRG/model_80000.d3 --traj_type CRG
