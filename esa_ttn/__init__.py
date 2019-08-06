import nltk
import sqlite3 as sql
from collections import defaultdict
from nltk.corpus import stopwords

import pickle
import pathlib as pl
import pandas as pd
import networkx as nx
import fuzzyset as fs

terms_for_articles_statement = """
SELECT terms.term, articles.article, tf_idf
FROM terms, term_article_score, articles
WHERE articles.article in ({}) AND articles.id = term_article_score.article_id AND terms.id = term_article_score.term_id 
ORDER BY tf_idf DESC;
                                    """


def map_tokens_to_composites(text_tokens, composites, composite_size: int = 2):
    composite_token_list = []
    i = 0
    while i < len(text_tokens):
        for s in range(composite_size, 0, -1):
            t_slice = text_tokens[i:i + s]
            composite_token = " ".join(t_slice)
            if composite_token in composites:
                composite_token_list.append(composite_token)
                i += s
                break
        else:
            composite_token_list.append(text_tokens[i])
            i += 1
    return composite_token_list


def batch(tokens, size):
    for i in range(0, len(tokens), size):
        t_chunk = tokens[i:i + size]
        yield t_chunk


class ESA_DB:

    def __init__(self, db_path, load_cached_fs=True):
        self.conn = sql.connect(db_path)
        self.db_path = db_path
        self.batchsize = 20
        self.load_cached_fs = load_cached_fs
        self.fuzzyset = None

    def get_article_vectors_for_terms(self, tokens):
        """
        Takes a list of tokens and returns a list of dicts with each dict
        representing a sparse tf_idf vector of the token with wikipedia articles as dimensions.
        Tokens need to be stemmed first using SnowballStemmer
        :param tokens:
        :return:
        """
        vectors = defaultdict(dict)
        for token_batch in batch(tokens, self.batchsize):
            param_placeholders = ", ".join(["?" for _ in range(len(token_batch))])
            statement = """SELECT terms.term, articles.article, tf_idf
                            FROM terms, term_article_score, articles
                            WHERE terms.term in ({}) AND terms.id = term_article_score.term_id AND articles.id = term_article_score.article_id
                            ORDER BY tf_idf DESC;
            """.format(param_placeholders)
            c = self.conn.cursor()
            c.execute(statement, token_batch)
            for term, article, tf_idf in c:
                vectors[term][article] = tf_idf
        return vectors

    def get_term_vectors_for_articles(self, tokens):
        """
        Takes a list of tokens and tries to extract wikipedia articles as tokens.
        :param tokens:
        :return:
        """
        c = self.conn.cursor()
        vectors = defaultdict(dict)
        for i in range(3, 0, -1):
            i_length_tokens = [" ".join(tokens[i2:i2 + i]) for i2 in range(0, (len(tokens) + 1 - i))]
            for token_batch in batch(i_length_tokens, self.batchsize):
                param_placeholders = ", ".join(["?" for _ in range(len(token_batch))])
                statement = terms_for_articles_statement.format(param_placeholders)
                c.execute(statement, token_batch)
                for term, article, tf_idf in c:
                    vectors[article][term] = tf_idf

        return vectors

    def get_term_vectors_for_articles_fuzzy(self, tokens, sim_threshold=0.8, gram_size=6, max_len_diff=5,
                                            use_levenshtein=True, composite_size=2):
        c = self.conn.cursor()
        if self.fuzzyset is None:
            fs_path = pl.Path(self.db_path + "_article_fs.pickle")
            if self.load_cached_fs and fs_path.exists():
                print("Loading Fuzzy Set from disk")
                with fs_path.open("rb") as fs_file:
                    f_s = pickle.load(fs_file)
                    self.fuzzyset = f_s
            else:
                print("Creating Fuzzy Set")
                all_articles_query = """SELECT articles.article
                                        FROM articles
                                        """
                c.execute(all_articles_query)
                f_s = fs.FuzzySet(gram_size_lower=gram_size, gram_size_upper=gram_size, use_levenshtein=False)
                i = 0
                for article, in c:
                    f_s.add(article)
                    i += 1
                    # if i % 10000 == 00:
                    #     print("Articles processed: {}".format(i))
                self.fuzzyset = f_s
                with fs_path.open("wb") as fs_file:
                    pickle.dump(f_s, fs_file)
                print("Finished creating Fuzzy Set")
        self.fuzzyset.use_levenshtein = use_levenshtein
        vectors = defaultdict(dict)
        token_article_mapping = {}
        for i in range(composite_size, 0, -1):
            i_length_tokens = [" ".join(tokens[i2:i2 + i]) for i2 in range(0, (len(tokens) + 1 - i))]
            matched_articles = []
            i_m = 1
            for i_t in i_length_tokens:
                i_m += 1
                # print("Processed Token: {}".format(i_m))
                match = self.fuzzyset.get(i_t)
                if match:
                    sim, word = match[0]
                    len_dif = abs(len(word) - len(i_t))
                    word_len = len(word) if len(word) <= 15 else 15
                    length_adjusted_threshold = sim_threshold + 0.15 * (word_len / 15)
                    condition = sim >= length_adjusted_threshold and len_dif <= max_len_diff
                    if condition:
                        token_article_mapping[i_t] = word
                        matched_articles.append(word)

            for token_batch in batch(matched_articles, self.batchsize):
                param_placeholders = ", ".join(["?" for _ in range(len(token_batch))])
                statement = terms_for_articles_statement.format(param_placeholders)
                c.execute(statement, token_batch)
                for term, article, tf_idf in c:
                    vectors[article][term] = tf_idf

        return vectors, token_article_mapping


def text_to_tokens(text, lang="english"):
    sw = set(stopwords.words(lang))
    t_tokens = nltk.word_tokenize(text, language=lang)
    t_tokens = [item.lower() for item in t_tokens if item not in sw and len(item) > 3 and not item.isdigit()]
    # if stem:
    #     stemmer = SnowballStemmer(language=lang)
    #     t_tokens = [stemmer.stem(item) for item in t_tokens]
    return t_tokens


def text_to_network_table(tokens, esa_db, window_size=10, map_tokens_to="terms"):
    mtt_vals = ["terms", "articles"]
    if map_tokens_to not in mtt_vals:
        raise Exception("map_tokens_to needs to be one of the values: {}".format(mtt_vals))
    print("-- Fetching Vectors")
    if map_tokens_to == "terms":
        stemmer = nltk.stem.SnowballStemmer(language=lang)
        tokens = [stemmer.stem(token) for token in tokens]
        vectors = esa_db.get_article_vectors_for_terms(tokens)
    elif map_tokens_to == "articles":
        vectors = esa_db.get_term_vectors_for_articles(tokens)
        tokens = map_tokens_to_composites(tokens, vectors)

    print("-- Creating Edge Table")
    i = 0
    possible_edge_table = []
    while True:
        last_token = tokens[i]
        last_token_vec = vectors[last_token]
        begin = max(0, i - window_size)
        rest_tokens = tokens[begin:i]
        for r_token in rest_tokens:
            if r_token == last_token:
                continue
            r_token_vec = vectors[r_token]

            if not r_token_vec or not last_token_vec:
                continue

            for k, v in r_token_vec.items():

                if k in last_token_vec:
                    conn_score = v * last_token_vec[k]
                    if map_tokens_to == "terms":
                        t_prefix = "Term:"
                        c_prefix = "Article:"
                    else:
                        t_prefix = "Article:"
                        c_prefix = "Term:"

                    possible_edge_table.append((t_prefix + last_token, t_prefix + r_token, c_prefix + k, conn_score))
        i += 1
        if i >= len(vectors):
            break

    columns = ["token_a", "token_b", "connecting_concept", "tfidf_score"]

    possible_edge_df = pd.DataFrame(possible_edge_table,
                                    columns=columns)

    unique_edge_dict = defaultdict(float)
    for tup in possible_edge_df.itertuples():
        t_a = tup.token_a
        t_b = tup.token_b
        conn_concept = tup.connecting_concept
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        unique_edge_dict[(t_a, t_b, conn_concept)] += tup.tfidf_score
    unique_edge_list = [(*k, v) for k, v in unique_edge_dict.items()]
    unique_edge_df = pd.DataFrame(unique_edge_list, columns=columns)
    return unique_edge_df.sort_values(by="tfidf_score", ascending=False)


def text_to_network_table_fuzzy_articles(tokens, esa_db, window_size=10, sim_threshold=0.8):
    vectors, token_article_mapping = esa_db.get_term_vectors_for_articles_fuzzy(tokens, sim_threshold=sim_threshold)
    article_token_mapping = {v: k for k, v in token_article_mapping.items()}
    # Transform single word tokens into composite tokens.
    tokens = map_tokens_to_composites(tokens, token_article_mapping)
    # Get all Articles that match to composite tokens in the text.
    token_mapping = {t: token_article_mapping[t] if t in token_article_mapping else t for t in tokens}
    tokens = list(token_mapping.values())
    possible_edge_table = []
    for i, last_token in enumerate(tokens):
        last_token_base = article_token_mapping.get(last_token, "no_base")
        last_token_vec = vectors[last_token]
        begin = max(0, i - window_size)
        rest_tokens = tokens[begin:i]
        for r_token in rest_tokens:
            if r_token == last_token:
                continue
            r_token_base = article_token_mapping.get(r_token, "no_base")
            r_token_vec = vectors[r_token]

            if not r_token_vec or not last_token_vec:
                continue

            for k, v in r_token_vec.items():

                if k in last_token_vec:
                    conn_score = v * last_token_vec[k]
                    t_prefix = "Article:"
                    c_prefix = "Term:"

                    possible_edge_table.append((t_prefix + last_token, t_prefix + r_token, c_prefix + k,
                                                last_token_base, r_token_base, conn_score))

    columns = ["token_a", "token_b", "connecting_concept", "token_a_base", "token_b_base", "tfidf_score"]

    possible_edge_df = pd.DataFrame(possible_edge_table,
                                    columns=columns)

    unique_edge_dict = defaultdict(float)
    for tup in possible_edge_df.itertuples():
        idx, t_a, t_b, conn_concept, a_base, b_base, tfidf_score = tup
        if t_a > t_b:
            t_a, t_b = t_b, t_a
            a_base, b_base = b_base, a_base
        unique_edge_dict[(t_a, t_b, conn_concept, a_base, b_base)] += tfidf_score
    unique_edge_list = [(*k, v) for k, v in unique_edge_dict.items()]
    unique_edge_df = pd.DataFrame(unique_edge_list, columns=columns)
    return unique_edge_df.sort_values(by="tfidf_score", ascending=False)


def edge_df_to_network(df, score_cutoff=50000, mode="connecting_concepts"):
    concept_graph = nx.Graph()
    for tup in df.itertuples():
        _, c_a, c_b, c_c, score = tup
        if score >= score_cutoff:
            if mode == "connecting_concepts":
                edges = [(c_a, c_c), (c_b, c_c)]
                for n_a, n_b in edges:
                    if concept_graph.has_edge(n_a, n_b):
                        concept_graph[n_a][n_b]["weight"] += 1
                    else:
                        concept_graph.add_edge(n_a, n_b, weight=1)
                concept_graph.nodes[c_c]["nodetype"] = "connecting_concept"
                concept_graph.nodes[c_a]["nodetype"] = "text_concept"
                concept_graph.nodes[c_b]["nodetype"] = "text_concept"
            elif mode == "direct_connection":
                if concept_graph.has_edge(c_a, c_b):
                    concept_graph[n_a][n_b]["weight"] += 1
                    concept_graph[n_a][n_b]["connecting_concepts"] = (
                            concept_graph[n_a][n_b]["connecting_concepts"] +
                            " "
                            + c_c)
                else:
                    concept_graph.add_edge(c_a, c_b, weight=1, connecting_concepts=c_c)

    return concept_graph


def edge_df_to_network_fuzzy(df, score_cutoff=50000, mode="connecting_concepts"):
    concept_graph = nx.Graph()
    for tup in df.itertuples():
        _, c_a, c_b, c_c, c_a_base, c_b_base, score = tup
        if score >= score_cutoff:
            if mode == "connecting_concepts":
                edges = [(c_a, c_c), (c_b, c_c)]
                for n_a, n_b in edges:
                    if concept_graph.has_edge(n_a, n_b):
                        concept_graph[n_a][n_b]["weight"] += 1
                    else:
                        concept_graph.add_edge(n_a, n_b, weight=1)
                concept_graph.nodes[c_c]["nodetype"] = "connecting_concept"
                concept_graph.nodes[c_a]["nodetype"] = "text_concept"
                concept_graph.nodes[c_b]["nodetype"] = "text_concept"
                concept_graph.nodes[c_a]["baseform"] = c_a_base
                concept_graph.nodes[c_b]["baseform"] = c_b_base

            elif mode == "direct_connection":
                if concept_graph.has_edge(c_a, c_b):
                    concept_graph[n_a][n_b]["weight"] += 1
                    concept_graph[n_a][n_b]["connecting_concepts"] = (
                            concept_graph[n_a][n_b]["connecting_concepts"] +
                            " "
                            + c_c)
                else:
                    concept_graph.add_edge(c_a, c_b, weight=1, connecting_concepts=c_c)

    return concept_graph


def filter_network_by(g: nx.Graph, type="eigenvector", cutoff_score=None):
    score_mapping = {
        "eigenvector": 0.2,
        "betweenness": 3,
        "core": 3,
    }

    if cutoff_score is None:
        cutoff_score = score_mapping[type]

    if len(g.nodes) == 0:
        return g
    if type == "eigenvector":
        ce = nx.eigenvector_centrality_numpy(g)
    if type == "degree":
        ce = nx.degree_centrality(g)
    if type == "betweenness":
        ce = nx.betweenness_centrality(g)
    if type == "core":
        ce = nx.core_number(g)
    g_f = g.copy()
    node_del_list = [n for n in g.nodes if ce[n] < cutoff_score]
    g_f.remove_nodes_from(node_del_list)
    return g_f


def main():
    r_path_base = pl.Path(data_dir) / "results"
    r_path_base.mkdir(exist_ok=True)
    db = ESA_DB(data_dir + "esa.db")
    with open(data_dir + "test_wiki.txt") as f:
        text = f.read()
        tokens = text_to_tokens(text, lang=lang)
    for t_m, thresh in [("articles", 5000), ("terms", 500000)]:
        r_path = r_path_base / (t_m + "_vectors")
        r_path.mkdir(exist_ok=True)
        graph_base = r_path / "graphs"
        graph_base.mkdir(exist_ok=True)
        print("Extracting Network Table")
        if t_m == "terms":
            edge_df = text_to_network_table(tokens, db, window_size=window_size, map_tokens_to=t_m)
            edge_df.to_csv(str(r_path / "edge_list.csv"))
            print("Filtering Graphs")
            c_n = edge_df_to_network(edge_df, thresh)
            nx.write_gml(c_n, str(graph_base / "concept_graph.gml"))
            f_type = "core"
            e_step = 1
            for i in range(0, 5):
                c_score = e_step * i
                c_n_f = filter_network_by(c_n, type=f_type, cutoff_score=c_score)
                graph_base.mkdir(exist_ok=True)
                nx.write_gml(c_n_f, str(graph_base / "concept_graph_filtered_{}.gml".format(c_score)))
        else:
            edge_df = text_to_network_table(tokens, db, window_size=window_size, map_tokens_to=t_m)
            edge_df.to_csv(str(r_path / "edge_list.csv"))
            c_n = edge_df_to_network(edge_df, thresh)
            nx.write_gml(c_n, str(graph_base / "concept_graph.gml"))
            edge_df_fuzzy = text_to_network_table_fuzzy_articles(tokens, db, window_size=window_size)
            edge_df_fuzzy.to_csv(str(r_path / "edge_list_fuzzy.csv"))
            c_n_fuzzy = edge_df_to_network_fuzzy(edge_df_fuzzy, thresh)
            nx.write_gml(c_n_fuzzy, str(graph_base / "concept_graph_fuzzy.gml"))
            print("Filtering Graphs")
            f_type = "core"
            e_step = 1
            for i in range(0, 5):
                c_score = e_step * i
                c_n_f = filter_network_by(c_n, type=f_type, cutoff_score=c_score)
                c_n_fuzzy_f = filter_network_by(c_n_fuzzy, type=f_type, cutoff_score=c_score)
                graph_base.mkdir(exist_ok=True)
                nx.write_gml(c_n_f, str(graph_base / "concept_graph_filtered_{}.gml".format(c_score)))
                nx.write_gml(c_n_fuzzy_f, str(graph_base / "concept_graph_fuzzy_filtered_{}.gml".format(c_score)))


if __name__ == "__main__":
    main()
