from openai import OpenAI
import pandas as pd
import argparse
from tqdm import tqdm
client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    data = client.embeddings.create(input = [text], model=model, dimensions=512).data[0].embedding
    return data


def load_dataset(dataset_name, split):
    df = pd.read_csv(f'{dataset_name}/Queried_{dataset_name}_all_models_clean_{split}.csv')
    return df

def query_embedding(df, dataset_name, model='text-embedding-3-small', split='train'):
    # Create a new column for embeddings
    df['embedding'] = None    
    # Use tqdm with apply method to show progress
    tqdm.pandas(desc="Getting embeddings")
    df['embedding'] = df['query_raw'].progress_apply(lambda x: get_embedding(x, model=model))
    
    # Save the dataframe with embeddings
    # dataset_name = df.name if hasattr(df, 'name') else args.dataset
    # for i in tqdm(range(len(df))):
    #     df.loc[i, 'embedding'] = get_embedding(df.loc[i, 'query_raw'], model=model)
    df.to_csv(f'{dataset_name}/embedded_{dataset_name}_all_models_clean_{split}.csv', index=False)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='HEADLINES')
    # for headline is small, for AGNEWS is large
    args.add_argument('--model', type=str, default='text-embedding-3-large')
    args.add_argument('--split', type=str, choices=['train', 'test'], default='train')
    args = args.parse_args()

    df = load_dataset(args.dataset, args.split)
    query_embedding(df, args.dataset, args.model, args.split)

    # python get_embedding.py --dataset OVERRULING --split train --model text-embedding-3-large
