import sys
import os

import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hebrew_text_simplification import text_simplification_pipeline

def main():
    dataset_file_name = r'C:\Users\bil\Desktop\text_simplification_plugin\ChromePlugin\hebrew_ts\human_eval_dataset_testing.csv'
    # dataset_file_name = r'C:\Users\bil\Desktop\text_simplification_plugin\ChromePlugin\hebrew_ts\human_eval_dataset.csv'
    df = pd.read_csv(dataset_file_name)
    simplifed_list = []
    for text in tqdm(df['text'], desc='simplifying'):
        simp = text_simplification_pipeline(text, word_sub=True, sentence_filter=False)
        simplifed_list.append(simp)
    df['simplified'] = simplifed_list
    summarized_list = []
    sentences_count_after_summ = []
    for text in tqdm(df['simplified'], desc='summarizing'):
        summ = text_simplification_pipeline(text, word_sub=False, sentence_filter=True)
        summarized_list.append(summ)
        sentences_count_after_summ.append([len(paragraph.split('. ')) for paragraph in summ.split('\n')])
    df['summary'] = summarized_list
    df['sentence counts after summarization'] = sentences_count_after_summ

    df.set_index('page id', inplace=True)
    df = df.astype({'title': 'string', 'text': 'string', 'simplified': 'string', 'summary': 'string'})
    df.to_csv('human_eval_system_dataset.csv', encoding='utf-8-sig')

if __name__ == '__main__':
    main()
