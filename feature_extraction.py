import nltk
import re
import itertools
import gensim
from tqdm import tqdm, tqdm_notebook
from nltk.stem import WordNetLemmatizer , SnowballStemmer
from nltk.corpus import treebank
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import json
import gensim
from gensim.models import Word2Vec
import pickle
import pandas as pd
import numpy as np

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('wordnet')

class Features:
    def __init__(self, product_descriptions):
        self.candidate_attributes = {}
        self.attribute_lexicon = json.load(open('new_new_attribute_lexicon.json', 'r'))
        self.product_descriptions = product_descriptions
        self.product_att_matrix = pd.DataFrame(index=list(self.product_descriptions.keys()), 
                                               columns=list(self.attribute_lexicon.keys()))
        for idx in list(self.product_att_matrix.index):
            for col in self.product_att_matrix.columns:
                self.product_att_matrix.loc[idx, col] = [0.001] * 300
        self.candidate_feature_POS =  [['JJ', 'JJ'], ['JJ', 'NN'],
                                      ['JJ', 'NNS'], ['JJ', 'RB'], ['JJ', 'VBD', 'JJ'], 
                                      ['JJ', 'VBG'], ['NN', 'CC', 'JJ'], ['NN', 'JJ'], 
                                      ['NN', 'NN'], ['NN', 'NN', 'VBD'], 
                                      ['NN', 'NNS'], ['NN', 'VBD'], ['NN', 'VBG'], ['NN', 'VBZ', 'NN', 'NN'], 
                                      ['NNS', 'NN'], ['VB', 'NN'], ['VBG', 'NN'], 
                                      ['VBZ', 'NN'], 'JJ', 'NN', 'NNS']
        self.frequency_counts = None
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)  
        self.pos = []
        self.attributes_similarity = {}
        self.similarities = {}
        
    def preprocess(self, sentence):
        sentence = sentence.lower()

        # Init the Snowball Stemmer
        stemmer = SnowballStemmer('english')

        # Stemmer Single Word
        sentence = " ".join(stemmer.stem(w) for w in sentence.split(" "))
        return sentence
    
    def get_similariteis_word2vec(self):
        lists = ['cushioning', 'outsole', 'midsole', 'insole',
              'Heel', 'Color', 'Shape', 'Upper', 'Fit', 
              'Weight', 'Density', 'Fixation', 'Toe Box',
              'collar', 'roll bars', 'gel pads', 'fasteners',
              'Achile notch', 'Permeability', 'Impact absorption',
              'Energy return', 'Stability', 'Flexibility',
              'Traction', 'Durability']
        for term in lists:
            term_dic = {}
            try:
                for i, j in self.word2vec.similar_by_word(term.lower()):
                    term_dic[i] = j
                self.attributes_similarity[term.lower()] = term_dic
            except:
                pass
        return 
    
            
    
    def find_similar_Word2Vec(self, sentence, product_key, preprocess = False):
        candidate_tags = []
        candidate_words = []
        if preprocess:
            sentence = self.preprocess(sentence)
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        self.pos.append(tags)
        for word, tag in tags:
            candidate_tags.append(tag)
            candidate_words.append(word)
        terms = [candidate_words[i:j] for i, j in itertools.combinations(range(len(candidate_words) + 1), 2)]
        i = 0
        for words in terms:
            w = " ".join(x for x in words)
            for key in list(self.attribute_lexicon.keys()):
                if w.lower() in self.attribute_lexicon[key]:
                    similarity_val = self.word2vec.similarity(w.lower(), key.lower())
                    self.product_att_matrix.loc[product_key, key] = similarity_val
                else:
                    try:
                        similarity_val = self.word2vec.similarity(w.lower(), key.lower())
                        if similarity_val > 0.5:
                            self.product_att_matrix.loc[product_key, key] = similarity_val
                    except:
                        pass           
                i += 1
        return 


    def find_similar_Vec(self, sentence, product_key, preprocess = False):
        for key in list(self.attribute_lexicon.keys()):
            for term in self.attribute_lexicon[key]:
                if term in sentence:
                    vectors = []
                    for word in term.split():
                        try:
                            vectors.append(list(self.word2vec.get_vector(word)))
                        except:
                            pass
                    vectors = np.mean(vectors, axis = 0)
                    self.product_att_matrix.loc[product_key, key] = vectors
        return 
    
    def get_attribute_matrix(self, preprocess = False):
        keys = list(self.product_descriptions.keys())
        for i in tqdm_notebook(range(len(keys))):
            key = keys[i]
            category = self.product_descriptions[key]
            model = category[0]
            [self.find_similar_Vec(x,key ,preprocess) for \
                                    x in model['description'].split(',')]
        return

    def get_similar_attributes(self, preprocess = False):
        self.get_similariteis_word2vec()
        keys = list(self.product_descriptions.keys())
        for i in tqdm_notebook(range(len(keys))):
            key = keys[i]
            category = self.product_descriptions[key]
            model = category[0]
            [self.find_similar_Word2Vec(x,key ,preprocess) for \
                                    x in model['description'].split(',')]
        return
                        
        
    def find_candidate_attributes(self, sentence, show = False, preprocess = False):
        candidate_tags = []
        candidate_words = []
        if preprocess:
            sentence = self.preprocess(sentence)
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        self.pos.append(tags)
        if show:
            print(tags)
        for word, tag in tags:
            candidate_tags.append(tag)
            candidate_words.append(word)
        combinations = [candidate_tags[i:j] for i, j in itertools.combinations(range(len(candidate_tags) + 1), 2)]
        terms = [candidate_words[i:j] for i, j in itertools.combinations(range(len(candidate_words) + 1), 2)]
        for tag, word in zip(combinations, terms):
            if tag in self.candidate_feature_POS:
                term = " ".join(w for w in word)
                if term not in self.candidate_attributes:
                    self.candidate_attributes[term] = {'frequency' : 1,
                                                           'pos' : tag}
                else:
                    self.candidate_attributes[term]['frequency'] += 1 
                    
    def get_candidate_attributes(self, show = False, num_example_show = 3 ,preprocess = False):
        i = 0
        for key in self.product_descriptions.keys():
            if i < num_example_show:
                show = True
            category = self.product_descriptions[key]
            for model in category:
                [self.find_candidate_attributes(x, show, preprocess) for x in model['description'].split(',')]
                show = False
            i += 1   
        return self.candidate_attributes, self.pos
    
    def get_frequent_attributes(self, support = 0.05):
        import collections
        self.frequency_counts = collections.Counter(self.candidate_attributes)
        for key, val in self.frequency_counts.items():
            self.frequency_counts[key] = self.frequency_counts[key]['frequency']/len(self.candidate_attributes)
        self.frequency_counts = { k:v for k,v in self.frequency_counts.items() if v >= support }
        return self.frequency_counts
    
    
    