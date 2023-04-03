from itertools import chain
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
import matplotlib.pyplot as plt
from pprint import pprint as display

from hebrew_ts.models import sentence_model
from hebrew_ts.preprocessing import Preprocessing


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

    # verteices_df = pd.DataFrame({'paragraph': sentences})
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
        print(similarity_matrix)
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
                     new_line_tok='\n',
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
        sentences = [Preprocessing.remove_dot(s) for s in sentences ]  
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
        if visualize:
            print('before filtering:')
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
        sorted_sentences_top_k = sorted([scored_sentences[i] for i in sentence_dict.keys()], key=lambda x: x[1], reverse=True)
        ranked_sentences_top_k = [sentence for sentence, _ in sorted_sentences_top_k]
        fixed_paragraphs.append(". ".join(ranked_sentences_top_k))
        if visualize:
            print('after filtering:')
            top_n_df = pd.DataFrame({'paragraph': ranked_sentences_top_k, 'pagerank score': [score for _, score in sorted_sentences_top_k]})
            formatted_res_df = top_n_df.style.format(formatter = None).set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: left;'},
                                                                {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.3em;'},
                                                                {'selector': 'td', 'props': 'text-align: right; vertical-align: top; width: 300px; direction: rtl;'},
                                                                ], overwrite=False)
            display(formatted_res_df)
    # output the summarized version
    return new_line_tok.join(fixed_paragraphs) #,len(sentences)