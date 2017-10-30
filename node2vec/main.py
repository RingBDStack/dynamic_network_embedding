'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
import logging
import time
from word2vec.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename="node2vec.log", filemode="a")


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='../output/bn.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='b.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.add_argument('--dynamic', dest='dynamic', default=True)
    parser.add_argument('--old_input)', dest='oldinput', default="../output/b.edgelist")
    parser.add_argument('--old_emb)', dest='oldemb', default="b.emb")
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(input):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(args.output)
    return


def find_changed_edge(old_g, new_g):
    old_edge = set([(u, old_g.G.edge[u].keys()[0]) for u in old_g.G.edge])
    new_edge = set([(u, new_g.G.edge[u].keys()[0]) for u in new_g.G.edge])

    vanish = old_edge - new_edge
    add = new_edge - old_edge

    print "-:", len(vanish), "+:", len(add)

    return vanish, add


def find_near_node(pair, G, deep=1):
    node_set = set([])
    new_node_set = set([])
    for i in range(len(pair)):
        node_set.add(pair[i])

    for node in node_set:
        for n in G.G.adj[node].keys():
            if n not in node_set:
                new_node_set.add(n)
    node_set |= new_node_set

    for i in range(deep):
        temp_set = set([])
        for node in new_node_set:
            for n in G.G.adj[node].keys():
                if n not in node_set:
                    node_set.add(n)
                    temp_set.add(n)
        new_node_set = temp_set
    return node_set


def train_vanish(walks, sent_edge_dict):
    walks = [map(str, walk) for walk in walks]
    if len(walks) > 0:
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, sent_edge_dict=sent_edge_dict)
        model.init_sims(replace=True)
        vec = {word: model.wv.syn0[model.wv.vocab[word].index] for word in model.wv.vocab}
        return vec
    else:
        return {}


def train_add(walks):
    walks = [map(str, walk) for walk in walks]
    if len(walks) > 0:
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
        model.init_sims(replace=True)
        vec = {word: model.wv.syn0[model.wv.vocab[word].index] for word in model.wv.vocab}
        return vec
    else:
        return {}


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if args.dynamic == True:
        print "dynamic"
        nx_G_old = read_graph(args.oldinput)
        G_old = node2vec.Graph(nx_G_old, args.directed, args.p, args.q)
        G_old.preprocess_transition_probs()

        nx_G_new = read_graph(args.input)
        G_new = node2vec.Graph(nx_G_new, args.directed, args.p, args.q)
        G_new.preprocess_transition_probs()

        print "load graph finish"
        print "old graph: nodes:", len(G_old.G.nodes()), "edges:", len(G_old.G.edges())
        print "new graph: nodes:", len(G_new.G.nodes()), "edges:", len(G_new.G.edges())

        vanish_edge, add_edge = find_changed_edge(G_old, G_new)

        vec = {}
        f = open(args.oldemb, "r")
        for line in f:
            node_vec = line.strip().split(" ")
            if len(node_vec) == args.dimensions + 1:
                vec[node_vec[0]] = np.array(map(str, node_vec[1:]))
        print "load vec finish"
        walk_vanish = []
        edge_count = 0
        sent_edge_dict = {}
        vanish_dict = {}

        for pair in vanish_edge:
            if pair[0] < pair[1]:
                if pair[0] in vanish_dict:
                    vanish_dict[pair[0]].add(pair[1])
                else:
                    vanish_dict[pair[0]] = {pair[1]}
            else:
                if pair[1] in vanish_dict:
                    vanish_dict[pair[1]].add(pair[0])
                else:
                    vanish_dict[pair[1]] = {pair[0]}
        near_node = set([])
        for pair in vanish_edge:
            near_node |= find_near_node(pair, G_old)
        print "near_node:", len(near_node)
        walks = G_old.simulate_walks(50, 5, nodes=list(near_node))
        print "gen corpus:", len(walks)
        for l in walks:
            p_idx = 0
            flag = 0
            for index in range(len(l) - 1):
                if l[index] < l[index + 1]:
                    k = l[index]
                    v = l[index + 1]
                else:
                    k = l[index + 1]
                    v = l[index]

                if k in vanish_dict and v in vanish_dict[k]:
                    if flag == 0:
                        flag = 1
                        p_idx = index
                    elif flag == 1:
                        edge = [l[p_idx], l[p_idx + 1]]
                        if k not in edge or v not in edge:
                            flag = 2
                            break
            if flag == 1:
                walk_vanish.append(l)
                sent_edge_dict[edge_count] = p_idx
                edge_count += 1

        print "vanish corpus:", len(walk_vanish)
        update_vec = train_vanish(walk_vanish, sent_edge_dict)

        for node in update_vec:
            if node in G_new.G.node:
                vec[node] = update_vec[node]
            else:
                del vec[node]
        print "update vec"

        near_node = set([])
        for pair in add_edge:
            near_node |= find_near_node(pair, G_new)
        walks = G_new.simulate_walks(50, 5, nodes=list(near_node))
        print "gen add corpus:", len(walks)
        update_vec = train_add(walks)
        for node in update_vec:
            vec[node] = update_vec[node]
        print "update vec"

        f = open(args.output, "a")
        f.truncate()
        for k in vec:
            f.write(k + " " + " ".join(map(str, vec[k])) + "\n")
    else:
        nx_G = read_graph(args.input)
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
        print "load graph finish"
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        print "gen corpus", len(walks)
        learn_embeddings(walks)
        print "finish"

if __name__ == "__main__":
    args = parse_args()
    # args.dynamic = eval(args.dynamic)
    print type(args.dynamic), args.input, args.oldemb, args.output
    a = time.time()
    main(args)
    b = time.time()
    logging.info(str(b - a))

