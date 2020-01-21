import csv
import string
import pickle
from collections import Counter
from itertools import chain

from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
import nltk

from stopwords import stopwords
from datasets import DATASET


class Question:
    """A class that represents Question objects"""
    
    __slots__ = ['id', 'title', 'body', 'tags', 'codes',
                 'has_code', 'has_link', 'has_math']
    
    def __init__(self, qid, title, body, tags):
        self.id = qid
        self.title = title
        self.body = body
        self.tags = tags
        
        self.codes = None
        
        self.has_code = False
        self.has_link = False
        self.has_math = False
        

class Preprocess:
    
    __slots__ = ['questions']
    
    def __init__(self, file_path):
        
        # Initializing the csv reader
        with open(file_path) as csvfile:
            reader = csv.reader(csvfile)
        
            # Retrieving questions and making Question object for each row
            # and building a list from them
            self.questions = self.remove_rare_tags(reader)
    
    
    def remove_rare_tags(self, reader):
        """Remove rare tags and respective empty questions"""

        # Skip the header row
        next(reader)
        questions = [Question(*row) for row in reader]

        # Removing tags that are occurred less than or equal to 50 times
        # and their associated questions
        all_tags_count = Counter(chain(*[q.tags.strip('<>').split('><')
                                        for q in questions]))
        tags_to_rem = set(tag for tag, count in all_tags_count.items()
                            if count <= 50)

        filtered_qs = []
        for q in questions:
            tags = q.tags.strip('<>').split('><')
            if set(tags) - tags_to_rem:
                q.tags = [tag for tag in tags if tag not in tags_to_rem]
                filtered_qs.append(q)

        return filtered_qs


    def check_code_link_math(self):
        """Check if body contains code, link or math formula"""
        
        for question in self.questions:
            if '<code>' in question.body:
                question.has_code = True
            if '</a>' in question.body:
                question.has_link = True
            if '$' in question.body:
                # Calculating the floor division of $ signs
                math_count = question.body.count('$') // 2
                if math_count:
                    question.has_math = True
    
    def remove_code_htmltags(self):
        """Removing code segments and HTML tags from question's body.
        Also saving the code in another field for later use.
        """
        
        for question in self.questions:
            soup = BeautifulSoup(question.body, 'html.parser')
            question.codes = ' '.join([code.get_text() for code
                                       in soup.find_all('code')])
            [s.extract() for s in soup('pre')]
            question.body = soup.get_text()
        
    def tokenize(self):
        """Tokenizing questions into words"""
        
        for question in self.questions:
            question.title = nltk.wordpunct_tokenize(question.title)
            question.body = nltk.wordpunct_tokenize(question.body)
            question.codes = nltk.wordpunct_tokenize(question.codes)

    def normalize(self):
        """Removing punctuation and lowercase conversion"""
        
        # Building a translate table for punctuation removal        
        punctnum_table = str.maketrans({c : None for c in string.punctuation})

        for question in self.questions:
            title_punct_removed = [token.translate(punctnum_table)
                                    for token in question.title]
            body_punct_removed = [token.translate(punctnum_table)
                                    for token in question.body]
            codes_punct_removed = [token.translate(punctnum_table)
                                    for token in question.codes]
            
            question.title = [token.lower() for token
                              in title_punct_removed if token]
            question.body = [token.lower() for token
                             in body_punct_removed if token]
            question.codes = [token.lower() for token
                              in codes_punct_removed if token]
            
    def remove_stopwords(self):
        """Removing stopwords from tokens"""
        
        for question in self.questions:
            question.title = [token for token in question.title 
                              if token not in stopwords]
            question.body = [token for token in question.body
                             if token not in stopwords]
            question.codes = [token for token in question.codes
                              if token not in stopwords]
    
    def stem(self):
        """Stemming tokens"""
        
        stemmer = PorterStemmer()
        
        for question in self.questions: 
            question.title = [stemmer.stem(token) for token in question.title]
            question.body = [stemmer.stem(token) for token in question.body]
            question.codes = [stemmer.stem(token) for token in question.codes]
            

def run_preprocesses(data_file):
    p = Preprocess(data_file)
    p.check_code_link_math()
    p.remove_code_htmltags()
    p.tokenize()
    p.normalize()
    p.remove_stopwords()
    p.stem()
    return p


def main():
    
    # Preprocessing and saving the data
    p = run_preprocesses(DATASET.data)
    with open(DATASET.root / 'preprocessed_data.pickle', 'wb') as file:
        pickle.dump(p.questions, file, protocol=pickle.HIGHEST_PROTOCOL)
            
    
if __name__ == '__main__':
    main()
