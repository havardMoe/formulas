import math

class BM25:
    def __init__(self,collection_length:int,avg_doc_length:int, doc_length:int,k1:float,b:float,base:int):
        self.collection_length = collection_length
        self.avg_doc_length = avg_doc_length
        self.k1 = k1 
        self.b = b
        self.base = base
        self.doc_length = doc_length

    def scorer(self,term_freq:int,collection_freq:int):
        idf = math.log((self.collection_length/collection_freq),self.base)
        top = (term_freq*(1+self.k1))
        bottom = (term_freq + (self.k1*(1-self.b+(self.b*(self.doc_length/self.avg_doc_length)))))
        score = (top/bottom) *idf
        return score


##### Examen2020 
doc1 = BM25(collection_length=1000,avg_doc_length=50,doc_length=25,k1=1.2,b=0.75,base = 10)
doc2 = BM25(collection_length=1000,avg_doc_length=50,doc_length=20,k1=1.2,b=0.75,base=10)

# Oppgave a)
term2_doc_1 = doc1.scorer(term_freq=0,collection_freq=50)
print(term2_doc_1)

term2_doc_2 =doc2.scorer(term_freq=3,collection_freq=50)
print(term2_doc_2)

##Oppgave b)

term_5_doc_1 = doc1.scorer(term_freq=10,collection_freq=100)
term_5_doc_2 = doc2.scorer(term_freq=1,collection_freq=100)

print(f'Doc1: {term2_doc_1+term2_doc_1+term_5_doc_1}')
print(f'Doc2: {term2_doc_2+term2_doc_2+term_5_doc_2}')
