import sys
import os

import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hebrew_text_simplification import text_simplification_pipeline

def main():
    dataset_file_name = r'C:\Users\bil\Desktop\text_simplification_plugin\ChromePlugin\hebrew_ts\human_eval_dataset_testing.csv'
    file_name = 'human_eval_system_dataset.csv'
    # file_name = 'human_eval_system_dataset_sample.csv'
    # dataset_file_name = r'C:\Users\bil\Desktop\text_simplification_plugin\ChromePlugin\hebrew_ts\human_eval_dataset.csv'
    # df = pd.read_csv(dataset_file_name).iloc[238:241,:]
    df = pd.read_csv(dataset_file_name)
    simplifed_list = []
    for r_index, row in tqdm(df.iterrows(), desc='simplifying', len=len(df)):
        try:
            simp = text_simplification_pipeline(row['text'], word_sub=True, sentence_filter=False)
            simplifed_list.append(simp)
        except RuntimeError:
            df.drop(index=r_index, inplace=True)
    df['simplified'] = simplifed_list
    df.to_csv(file_name, encoding='utf-8-sig')

    summarized_list = []
    sentences_count_after_summ = []
    for r_index, row in tqdm(df.iterrows(), desc='summarizing', len=len(df)):
        try:
            summ = text_simplification_pipeline(row['simplified'], word_sub=False, sentence_filter=True)
            summarized_list.append(summ)
            sentences_count_after_summ.append([len(paragraph.split('. ')) for paragraph in summ.split('\n')])
        except RuntimeError:
            df.drop(index=r_index, inplace=True)
    df['summary'] = summarized_list
    df['sentence counts after summarization'] = sentences_count_after_summ
    df.to_csv(file_name, encoding='utf-8-sig')

    summ_no_simp_list = []
    for r_index, row in tqdm(df.iterrows(), desc='only summary', len=len(df)):
        try:
            summ = text_simplification_pipeline(row['text'], word_sub=False, sentence_filter=True)
            summ_no_simp_list.append(summ)
        except RuntimeError:
            df.drop(index=r_index, inplace=True)
    df['only summary'] = summ_no_simp_list

    df.set_index('page id', inplace=True)
    df = df.astype({'title': 'string', 'text': 'string', 'simplified': 'string', 'summary': 'string', 'only summary': 'string'})
    df.to_csv(file_name, encoding='utf-8-sig')
    

if __name__ == '__main__':
    main()
