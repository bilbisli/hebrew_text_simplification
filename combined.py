import time
from abc import ABC, abstractmethod
from math import ceil, floor
import re
from string import punctuation
from functools import lru_cache
from collections import defaultdict
from kneed import KneeLocator
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx
import pandas as pd
from pandas import option_context
from itertools import chain
import wordfreq
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
from transformers import logging
from transformers import pipeline
from transformers import AutoTokenizer, BertForTokenClassification, BertForMaskedLM
from sentence_transformers import SentenceTransformer, util
from evaluate import load
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score
#import kneed
from scipy.spatial.distance import cdist
import os
from pprint import pprint as display


class Preprocessing():
  """ This class is a hebrew preprocess """
  def __init__(self, sentence):
    self.sentence = sentence

  def remove_punctuation(self):
    """ This method removes punctuation marks """
    self.sentence = re.sub(r'[^\w\s]', '', self.sentence)
  
  def remove_english_letters(self):
    self.sentence = re.sub("\n", " ", self.sentence) 
    self.sentence = re.sub("\xa0", " ", self.sentence) 
    self.sentence = re.sub("\r", " ", self.sentence) 
    self.sentence = re.sub("\r", " ", self.sentence) 
    self.sentence = re.sub("  ", " ", self.sentence)
  
  def remove_text_between_parentheses(self):
    """ This method removes text that appears between Parentheses """
    self.sentence = re.sub("[\(\[].*?[\)\]]", "", self.sentence)
   
  def preprocess(self):
    """ This method runs all the reqirements for hebrew data preprocessing. """
    self.remove_text_between_parentheses()
    self.remove_english_letters()

    return self.sentence

def remove_dot(sentence):
    return re.sub("\.$", "", sentence.strip())

def get_mask_model(model_name="imvladikon/alephbertgimmel-base-512"):
    return BertForMaskedLM.from_pretrained(model_name)

def get_ner_model(model_name="jony6484/alephbert-base-finetuned-ner-v2"):
    return BertForTokenClassification.from_pretrained(model_name)

def get_sentence_model(model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    return SentenceTransformer(model_name)

def get_tokenizer(model_name, model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    NEW_LINE_TOKEN = '<NL>'
    num_added_toks = tokenizer.add_tokens(NEW_LINE_TOKEN)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer

def create_mask_model_pipeline():
    model = get_mask_model()
    tokenizer = get_tokenizer("imvladikon/alephbertgimmel-base-512", model)

    return pipeline("fill-mask", model=model, tokenizer=tokenizer)

def create_ner_model_pipeline():
    model = get_ner_model()
    tokenizer = get_tokenizer("jony6484/alephbert-base-finetuned-ner-v2", model)
    
    return pipeline("ner", model=model, tokenizer=tokenizer)

NEW_LINE_TOKEN = '<NL>'
new_line_tok = '\n'
score_threshold = 0.32


mask_model = get_mask_model()
tokenizer = get_tokenizer("imvladikon/alephbertgimmel-base-512", mask_model)

ner_model = get_ner_model()
NER_TOK = get_tokenizer("jony6484/alephbert-base-finetuned-ner-v2", ner_model)

sentence_model = get_sentence_model()

model = create_mask_model_pipeline()
NER = create_ner_model_pipeline()

@lru_cache(maxsize=10)
def get_common_words(top_n=27000, lang='he'):
    return wordfreq.top_n_list(lang, top_n)

@lru_cache(maxsize=None)
def get_word_freq(word, lang='he'):
    return wordfreq.word_frequency(word, lang)

@lru_cache(maxsize=10)
def get_top_n_frequency(top_n=27000, lang='he'):
    top_common_words = get_common_words(top_n, lang)
    last_common_word = top_common_words[-1]
    top_n_freq = get_word_freq(last_common_word, lang)
    return top_n_freq


def is_number(text):
    return re.match('[+-]?(\d+)?\.?\d+', text) is not None

def word_split_ner_mask(text, tokenizer, ner_model, mask_model):
    tokenized_text = tokenizer(text, return_offsets_mapping=True)
    ners = ner_model(text)

    ner_indexes = [enr_ent['index'] for enr_ent in ners]
    ner_mask = [index in ner_indexes for index in range(len(tokenized_text['input_ids']))]


    added_tokens, true_word_split, offset_ner_mask = [], [], []
    for idx, word_id in enumerate(tokenized_text.word_ids()):
        if word_id is not None:
            start, end = tokenized_text.word_to_tokens(word_id)
            if start == end - 1:
                token_range = (start,)
            else:
                token_range = (start, end-1)
            if len(added_tokens) == 0 or added_tokens[-1] != token_range:
                decoded_word = tokenizer.decode(tokenized_text['input_ids'][token_range[0]:token_range[-1]+1], skip_special_tokens=True)
                added_tokens.append(token_range)
                true_word_split.append(decoded_word)
                offset_ner_mask.append(ner_mask[idx])

    return true_word_split, offset_ner_mask

def mask_and_replace(text_list, index, tokenizer, mask_token='[MASK]', score_threshold=0.23, new_line_tok=NEW_LINE_TOKEN, is_neighbour=False):
    new_text_list = text_list[:]
    new_text_list[index] = mask_token
    
    sen = tokenizer.convert_tokens_to_string(list(filter(lambda w: w != NEW_LINE_TOKEN, new_text_list)))
    candidates = model(sen)
    new_text_list[index] = text_list[index]

    if candidates[0]['score'] >= score_threshold \
    and (is_neighbour 
         or is_not_word(candidates[0]['token_str']) 
         or get_word_freq(candidates[0]['token_str']) > get_word_freq(text_list[index])):
            
            new_text_list[index] = candidates[0]['token_str']
   

    return new_text_list

def is_not_word(word):
    return is_number(word) or word in list(punctuation) or word == NEW_LINE_TOKEN

def find_mask_index(text_list, mask_exculsion, check_frequency=True):
    most_common_word_freq = get_top_n_frequency()
    mask_indices = []
    for i, word in enumerate(text_list):
     
        if mask_exculsion[i] \
                or is_not_word(word) \
                or check_frequency \
                and (get_word_freq(word) >= most_common_word_freq):

            mask_indices.append(False)
        else:
            mask_indices.append(True)

    return mask_indices

def unspace_decimal_numbers(text):
    return re.sub(r"(\d+) *(\.) *(\d+)", r"\1\2\3", text)

def simplify_words(text, index_list=None, tokenizer=None, ner_model=None, mask_model=None, score_threshold=0.23, check_frequency=True, neighbours_threshold=0.7):
    if index_list is None:
        if not all((tokenizer, ner_model, mask_model)):
            raise ValueError('if index list is not provided a tokenizer, ner model and mask model need to be provided')
        text_list, ner_mask = word_split_ner_mask(text, tokenizer, ner_model, mask_model)
        index_list = find_mask_index(text_list, ner_mask, check_frequency=check_frequency)
    words_bar = tqdm(enumerate(index_list), desc='word masking', leave=False, total=len(index_list), disable=len(index_list)==0)

    neighb = False

    for i, mask_word in words_bar:

        if mask_word:
            prev_word = text_list[i]
            text_list = mask_and_replace(text_list, i, tokenizer, score_threshold=score_threshold)

            if prev_word != text_list[i] and i + 1 < len(index_list) and not ner_mask[i + 1] and not index_list[i + 1] and not is_not_word(text_list[i + 1]):
                text_list = mask_and_replace(text_list, i + 1, tokenizer, score_threshold=neighbours_threshold, is_neighbour=True)
                neighb = True
                
        else:
            neighb = False

    tokens_to_text = tokenizer.convert_tokens_to_string(text_list)
    fixed_text = unspace_decimal_numbers(tokens_to_text)
    return fixed_text

def read_article(text, preprocess=True):        
    sentences = []
    new_sent = []        
    sentences = sent_tokenize(text)
    if preprocess:
        for sentence in sentences:
            processor = Preprocessing(sentence)       
            new_sent.append(processor.preprocess())
        return new_sent
    return sentences

def sentence_similarity(sentences_embeddings, top_k=None):
    if top_k is None:
        top_k = len(sentences_embeddings)
    return util.paraphrase_mining_embeddings(torch.tensor(sentences_embeddings), corpus_chunk_size=len(sentences_embeddings), top_k=top_k)

def get_sentence_embeddigns(sentences, model=sentence_model):
    embeddings = model.encode(sentences)
    return embeddings

def build_similarity_matrix(sentences_similarities):

    all_nodes = set(chain.from_iterable([(i, j) for _, i, j in sentences_similarities]))
    similarity_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for sentence_similarity_ in sentences_similarities:
        score, idx1, idx2 = sentence_similarity_
        similarity_matrix[idx1][idx2] = score
    return similarity_matrix

def group_to_clusters(clusters, lst):
    clustered_list = [[] for i in set(clusters)]

    for i, cluster in enumerate(clusters):
        clustered_list[cluster].append(lst[i])

    return clustered_list

def find_clusters_by_breaking_point(data_embeddings, visualize=False):

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    cluster_range = range(1, len(data_embeddings)+1)
    
    for k in cluster_range:
        # Building and fitting the model
        kmeanModel = KMeans(init='k-means++', n_clusters=k, random_state=0).fit(data_embeddings)
        kmeanModel.fit(data_embeddings)
        inertias.append(kmeanModel.inertia_)

    if visualize:
        plt.plot(cluster_range, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertias')
        plt.title('The Elbow Method using inertias')
        plt.show()

    kneedle = KneeLocator(cluster_range, inertias, S=0.45, curve="convex", direction="decreasing")

    if kneedle.knee is None:
        values = sentence_similarity(data_embeddings, top_k=None)
        if min(values)[0] < 0.5:
            return len(data_embeddings)
        else:
            return 1

    return int(kneedle.elbow)

def visualize_clusters(sentences, sentence_embeddings, clusters):
    # Apply PCA to reduce dimensionality
    pca = PCA()
    reduced_embeddings = pca.fit_transform(sentence_embeddings)

    # Plot the clusters
    dim = 1
    if len(sentences) == 1:
        dim = 0
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, dim], c=clusters, cmap='rainbow')

    # Add the sentences color-coded by their respective cluster to the visualization
    num_clusters = len(set(clusters))
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence[::-1], (reduced_embeddings[i, 0], 
                                reduced_embeddings[i, dim]),
                                color=plt.cm.rainbow(clusters[i] / num_clusters))
    plt.show()

def cluster_sentences_from_embeddings(sentences, sentence_embeddings, num_clusters=None, visualize=False):

    if len(sentences) == 1:
        clusters = [0]
        num_clusters = 1
    else:
        if num_clusters is None:
            num_clusters = find_clusters_by_breaking_point(sentence_embeddings, visualize=visualize)                                                   

        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(sentence_embeddings)

    # Group sentences into clusters
    clustered_sentences = [[] for i in range(num_clusters)]
    for i, cluster in enumerate(clusters):
        clustered_sentences[cluster].append(sentences[i])
    
    if visualize:
        visualize_clusters(sentences, sentence_embeddings, clusters)
    
    return clustered_sentences, clusters

def cluster_sentences(sentences, num_clusters=None, visualize=False):
    sentence_embeddings = get_sentence_embeddigns(sentences)
    clustered_sentences, clusters = cluster_sentences_from_embeddings(sentences, sentence_embeddings, num_clusters, visualize=visualize)
    
    return clusters

def display_sentence_simalirity_graph(sentence_similarity_graph):
    plt.figure(3,figsize=(8,8))
    #labels = nx.get_edge_attributes(sentence_similarity_graph,'weight')
    # labels = {n: sentence_similarity_graph.nodes[n]['weight'] for n in sentence_similarity_graph.edges}
    labels = {n: sentence_similarity_graph.edges[n]['weight'] for n in sentence_similarity_graph.edges}
    # display(labels)
    # , labels={i: s for i, s in enumerate(sentences)}
    pos = nx.spring_layout(sentence_similarity_graph)
    pos_higher = {}

    for k, v in pos.items():
        if(v[1]>0):
            pos_higher[k] = (v[0], v[1]+0.1)
        else:
            pos_higher[k] = (v[0], v[1]-0.1)

    verteices_df = pd.DataFrame({'paragraph': sentences})
    nx.draw(sentence_similarity_graph, with_labels=True, pos=nx.circular_layout(sentence_similarity_graph))
 
    # nx.draw_networkx_labels(sentence_similarity_graph, pos_higher, labels={i: f'{i}. {s[::-1]}' for i, s in enumerate(sentences)})
    nx.draw_networkx_edge_labels(sentence_similarity_graph, pos=nx.circular_layout(sentence_similarity_graph), edge_labels=labels)
    # top_k = len(sentences)-5
    # plt.savefig(f'graphs/paragraph_{parg}_{top_k}.png')
    plt.show()
    plt.clf()

def text_rank(text_embeddings, top_k=None, display_matrix=False, display_graph=False, edge_filter=None):
    if top_k is None:
        top_k = len(text_embeddings)
    sentence_similarities = sentence_similarity(text_embeddings, top_k=top_k)
    similarity_matrix = build_similarity_matrix(sentence_similarities)
    if display_matrix:
        display(similarity_matrix)
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    if edge_filter is not None:
        drop_n_edges = edge_filter(sentence_similarity_graph.number_of_edges())
        sorted_edges = sorted(sentence_similarity_graph.edges.data('weight'), key=lambda x: x[2])
   
        # remove the first k edges from the sorted list
        edges_to_remove = sorted_edges[:drop_n_edges]
 
        # remove the lowest edges by weight
        sentence_similarity_graph.remove_edges_from(edges_to_remove)
    # # iterate over all the edges of the graph
    # for u, v, w in sentence_similarity_graph.edges(data='weight'):
    #     if w < 0:
    #         # if the weight is negative, take its absolute value
    #         sentence_similarity_graph[u][v]['weight'] = abs(w)

    if display_graph:
        display_sentence_simalirity_graph(sentence_similarity_graph)

    is_converged = False
    tolerance = 1e-06
    while not is_converged:

        try:
            scores = nx.pagerank(sentence_similarity_graph, max_iter=len(text_embeddings), tol=tolerance)
            is_converged = True
        except nx.exception.PowerIterationFailedConvergence as e:
            tolerance *= 10


    if len(scores) == 0 or len(text_embeddings) == 1:
        scores = {0: 1.0}
    return scores

class EliminationStrategy(ABC):
    def __init__(self, sentences, scores_dict, top_n_to_leave, clusters=None):
        self.sentences = sentences
        self.scores_dict = scores_dict
        self.top_n_to_leave = top_n_to_leave
        self.clusters = [] if clusters is None else clusters
        self.sentence_dict = {i: sentence for i, sentence in enumerate(self.sentences)}

    @property
    @abstractmethod
    def get_n_clusters(self):
        pass

    @property
    def grouped_clusters(self):
        clustered_sentence_indices = [set() for s in set(self.clusters)]
        for i, cluster_index in enumerate(self.clusters):
            clustered_sentence_indices[cluster_index].add(i)
        return clustered_sentence_indices

    @property
    @abstractmethod
    def elimination_indices(self):
        pass

    def eliminate(self, clusters):
        self.clusters = clusters
        for i in self.elimination_indices():
            del self.sentence_dict[i]
        return self.sentence_dict

class EliminateLowestScoreInCluster(EliminationStrategy):
    def get_n_clusters(self):
        return len(self.sentences) - self.top_n_to_leave

    def elimination_indices(self):
        eliminiation_idxs = set()
        for cluster in self.grouped_clusters:
            eliminiation_idxs.add(min(cluster, key=self.scores_dict.get))
        return eliminiation_idxs


class EliminateAllButHighestScoreInCluster(EliminationStrategy):
    def get_n_clusters(self):
        return self.top_n_to_leave
        
    def elimination_indices(self):
        eliminiation_idxs = set()
        for cluster in self.grouped_clusters:
            cluster.remove(max(cluster, key=self.scores_dict.get))
            eliminiation_idxs.update(cluster)
        return eliminiation_idxs

def generate_summary(text, 
                     top_n_func=None, 
                     strategy=None, 
                     preprocess=True, 
                     new_line_tok=NEW_LINE_TOKEN,
                     visualize=False,
                     edge_filter=None):
    fixed_paragraphs = []
    # read text and tokenize 
    paragraphs = [p for p in text.split(new_line_tok) if p != '' and not p.isspace()]
    for paragraph in paragraphs:
        summarize_text = []
  
        sentences = [s for s in read_article(paragraph, preprocess=preprocess) if s != '' and not s.isspace()]
  
        if len(sentences) <= 2:
            fixed_paragraphs.append(sentences[0])
            continue
        sentences = [ remove_dot(s) for s in sentences ]  
        sentence_embeddings = get_sentence_embeddigns(sentences)
        top_n = None
        if top_n_func is not None:
            top_n = top_n_func(len(sentences))
        # number of sentences to remove
        # remove_k = len(sentences) - top_n

        # get scores
        sentences_text_rank_dict = text_rank(sentence_embeddings, 
                                             edge_filter=edge_filter,
                                             display_matrix=visualize, 
                                             display_graph=visualize)
        

        scored_sentences = [(sentence, sentences_text_rank_dict[i]) \
                            for i, sentence in enumerate(sentences)]
        sorted_sentences = sorted(scored_sentences,
                                  key=lambda x: x[1], reverse=True)
        top_n_df = pd.DataFrame({'paragraph': [sentence for sentence, _ in sorted_sentences], 'pagerank score': [score for _, score in sorted_sentences]})
        formatted_res_df = top_n_df.style.format(formatter=None).set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: left;'},
                                                            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.3em;'},
                                                            {'selector': 'td', 'props': 'text-align: right; vertical-align: top; width: 300px; direction: rtl;'},
                                                            ], overwrite=False)
        display(formatted_res_df)

        if strategy is None:
            strategy = EliminateAllButHighestScoreInCluster
        active_strategy = strategy(sentences, 
                            scores_dict=sentences_text_rank_dict, 
                            top_n_to_leave=top_n)
        # remove lowest score from each cluster
        remove_k = None
        if top_n_func is not None:
            remove_k = active_strategy.get_n_clusters()
       
        ## get as many clusters as the sentences to remove
        _, clusters = cluster_sentences_from_embeddings(sentences, 
                                                        sentence_embeddings, 
                                                        num_clusters=remove_k,
                                                        visualize=visualize)
        #visualize_clusters(sentences, sentence_embeddings, clusters)
        sentence_dict = active_strategy.eliminate(clusters)
        #visualize_clusters([sentences[i] for i in sentence_dict], [sentence_embeddings[i] for i in sentence_dict], [clusters[i] for i in sentence_dict])
        # build a sorted list of sentences by score
        scored_sentences = [(sentence, sentences_text_rank_dict[i]) \
                            for i, sentence in enumerate(sentences)]
        sorted_sentences = sorted(scored_sentences,
                                  key=lambda x: x[1], reverse=True)
        ranked_sentences_top_k = [sentence for sentence, _ in sorted_sentences]
        fixed_paragraphs.append(". ".join(ranked_sentences_top_k))

        top_n_df = pd.DataFrame({'paragraph': ranked_sentences_top_k, 'pagerank score': [score for _, score in sorted_sentences]})
        formatted_res_df = top_n_df.style.format(formatter = None).set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: left;'},
                                                            {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.3em;'},
                                                            {'selector': 'td', 'props': 'text-align: right; vertical-align: top; width: 300px; direction: rtl;'},
                                                            ], overwrite=False)
        display(formatted_res_df)
    # output the summarized version
    return new_line_tok.join(fixed_paragraphs) #,len(sentences)

def magic(text):
    simp_list = []
    summ_list = []
    paragraphs = [p for p in text.split(new_line_tok) if p != '' and not p.isspace()]
    for paragraph in paragraphs:
        simp = simplify_words(paragraph, tokenizer=NER_TOK, ner_model=NER, mask_model=model, score_threshold=score_threshold)
        
        summ = generate_summary(simp, top_n_func=None, new_line_tok=NEW_LINE_TOKEN, visualize=False)
        simp_list.append(simp)
        summ_list.append(summ)

    simp_text = new_line_tok.join(simp_list)
    summ_text = new_line_tok.join(summ_list)
    
    return summ_text
