from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess    

def tidy_sentence(sentence, vocabulary):
    return [word for word in simple_preprocess(sentence) if word in vocabulary]    

def compute_sentence_similarity(sentence_1, sentence_2, model_wv):
    vocabulary = set(model_wv.index2word)    
    tokens_1 = tidy_sentence(sentence_1, vocabulary)    
    tokens_2 = tidy_sentence(sentence_2, vocabulary)    
    return model_wv.n_similarity(tokens_1, tokens_2)

wv = KeyedVectors.load('model.wv', mmap='r')
sim = compute_sentence_similarity('this is a sentence', 'this is also a sentence', wv)
print(sim)