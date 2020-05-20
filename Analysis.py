#Vamsi Gajjela
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


def parse_data(document, dictionary: Dict[str, List[str]]) -> None:
    """
    Goes through a document of reviews and assign the content a review that is
    either 'pos' or 'neg' based on if the value in the text is 1 or 0 respectively
    and adds on the review and content to a given dictionary (modify's it)
    Precondition: the reviews in the doc are formatted as "sentence \t score \n"
    """
    for review in document:
        if review != '\n':
            review = review.split('\t')
            rating = review[1].strip('\n')
            dictionary['Review'].append('pos' if rating == '1' else 'neg')
            dictionary['Content'].append(review[0].strip('  '))


docs = (open('imdb_labelled.txt', 'r'), open('yelp_labelled.txt', 'r'), open('amazon_cells_labelled.txt', 'r'))
# Goes through databases from imdb yelp and amazon cells

data = {'Content': [], 'Review': []}
# Sets up dictionary that will contain the contents and reviews of data sets

for doc in docs:
    parse_data(doc, data)
    # parses through each document given

vector = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
# Initializes vecotorizer using TFID as it most efficient way to sort data with given parameters
# https://medium.com/@vasista/sentiment-analysis-using-svm-338d418e3ff1

points = vector.fit_transform(data['Content'])

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(points, data['Review'])

print('This is a Sentiment Analysis program that works using a linear SVM model')
print('Type in your review or just hit enter to exit')

user_input = input("Enter the review you want to test: ")

while user_input != '':
    user_input = vector.transform([user_input])
    print('The review is', classifier_linear.predict(user_input)[0], '\n' + '-' * 80)
    user_input = input("Enter the review you want to test: ")

print('\nThanks for using this program!')
