from flask import Flask, render_template, jsonify, abort, make_response, request, Response, url_for
from flask.views import MethodView
import socket
import os
import pandas as pd
import json
import subprocess
import datetime
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import numpy as np
import csv
import imdbreviewscraper


APP = Flask(__name__, template_folder="../templates", static_folder="../static")

ALLOWED_EXTENSIONS = set(['tsv'])


train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
#test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'testData.tsv'), header=0, delimiter="\t", \quoting=3 )

print 'The first review is:'
print train["review"][0]

#raw_input("Press Enter to continue...")


print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
#nltk.download()  # Download text data sets, including stop words

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange( 0, len(train["review"])):
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


# ****** Create a bag of words from the training set
#
print "Creating the bag of words...\n"


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                         tokenizer = None,    \
                         preprocessor = None, \
                         stop_words = None,   \
                         max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

# ******* Train a random forest using the bag of words
#
print "Training the random forest (this may take a while)..."


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )



@APP.errorhandler(400)
def not_found(error):
    """
    Gives error message when any bad requests are made.
    Args:
        error (string): The first parameter.
    Returns:
        Error message.
    """
    print error
    return make_response(jsonify({'error': 'Bad request'}), 400)


@APP.errorhandler(404)
def not_found(error):
    """
    Gives error message when any invalid url are requested.
    Args:
        error (string): The first parameter.
    Returns:
        Error message.
    """
    print error
    return make_response(jsonify({'error': 'Not found'}), 404)



@APP.route("/")
def index():
    """
    Gets to the index page of VDM when http://localhost:8000/ is hit.
    """
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def predict_sentiment(test):
    clean_test_reviews = []
    reviewstext = []
    with open('scraped.csv','rt') as csvfile1:
        reader = csv.reader(csvfile1)
        headers = next(reader, None)
        for row in reader:
            reviewstext.append(row[1])
            clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist((row[1]).decode('utf-8', 'ignore'), True)))
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    return {"review":reviewstext, "sentiment":result}
    #output = pd.DataFrame( data={"review":reviewstext, "sentiment":result} )
    #output.to_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), index=False, quoting=3, quotechar='', sep='\t')
    #output.to_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), index=False, quoting=csv.QUOTE_NONE, quotechar='', sep='\t')

    #print test
    # Create an empty list and append the clean reviews one by one
    '''
    print test["review"][0]
    print "type is ", type(test["review"][0])
    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,len(test["review"])):
        if type(test["review"][i]) is not float:
            try:
                clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist((test["review"][i]).decode('utf-8', 'ignore'), True)))
            except:
                #print "not possible"
                clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i])))
        else:
            clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist("Info not available")))
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)
    print type(result)
    print result
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"review":test["review"], "sentiment":result} )
    print output
    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), index=False, quoting=3, quotechar='', sep='\t')
    print "Wrote results to Bag_of_Words_model.csv"
    '''

def bag_of_words_classifier():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )

    print 'The first review is:'
    print train["review"][0]

    raw_input("Press Enter to continue...")


    print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...'
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


    # ****** Create a bag of words from the training set
    #
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # ******* Train a random forest using the bag of words
    #
    print "Training the random forest (this may take a while)..."


    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )



    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
#    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output = pd.DataFrame(data={"review":test["review"], "sentiment":result})
    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'analysis.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"

class TestingAPI(MethodView):
    @staticmethod
    def get():
        return jsonify({'status':'1'})

class DownloadReviewsAPI(MethodView):
    @staticmethod
    def post():
        #url = request.json['review_url']
        url = request.json['reviewurl']
        #print url
        #url = request.json['reviewurl']
        #print url

        imdbreviewscraper.scrape_reviews(url)
        time.sleep(4)
        #test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'scraped.tsv'), header=0,delimiter="\t", quoting=3)
        #test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'scraped.csv'), header=0, quoting=3)
        #test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'scraped.csv'), header=0, engine='python', quoting=csv.QUOTE_NONE, encoding='utf-8')
        #test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'scraped.csv'), header=0)
        #result = predict_sentiment(test)
        result = predict_sentiment("ab")
        time.sleep(4)
        positive_sentiments = []
        negative_sentiments = []
        for i in range(len(result["review"])):
            try:
                if (result["sentiment"])[i] in ['1', 1]:
                    positive_sentiments.append(result["review"][i])
                elif (result["sentiment"])[i] in ['0', 0]:
                    negative_sentiments.append(result["review"][i])

            except:
                print "error"

        response = {}
        response['negative'] = negative_sentiments
        response['positive'] = positive_sentiments
        return jsonify(response)
        '''
        with open('Bag_of_Words_model.csv', 'rt') as analysisfile:
            reader = csv.reader(analysisfile)
            headers = next(reader, None)
            for row in reader:
                print row
                row = (row[0]).replace('\n', '')
                row = row.replace('\t', '')
                sentiment = (row)[-1]
                each_review = (row)[:-1]
                try:
                    if sentiment == '1':
                        positive_sentiments.append(each_review)
                    elif sentiment == '0':
                        negative_sentiments.append(each_review)
                except:
                    print "error"
        response = {}
        response['negative'] = negative_sentiments
        response['positive'] = positive_sentiments

        #return jsonify(response)
        return jsonify(response)

        #
        #     for row in reader:
        #         if row[1] == '1':
        #             positive_sentiments.append(row[0])
        #         elif row[1] == '0':
        #             negative_sentiments.append(row[0])
        # response = {}
        # response['negative'] = negative_sentiments
        # response['positive'] = positive_sentiments
        #
        # #return jsonify(response)
        # return jsonify(response)
        '''

class SegregatePositiveNegativeAPI(MethodView):
    @staticmethod
    def get():
        with open('analysis.csv','rt') as analysisfile:
            reader = csv.reader(analysisfile)
            headers = next(reader, None)


class DownloadAnalysisAPI(MethodView):

    @staticmethod
    def get():
        output_csv = open('analysis.csv')
        return Response(
            output_csv,
            mimetype="xml/file",
            headers={"Content-disposition":
            "attachment; filename=analysis.csv"})


class UploadTrainingSetAPI(MethodView):

    @staticmethod
    def post(uploadfile):
        file = request.files['file']
        if file and allowed_file(file.filename):
            train = pd.read_csv(file, header=0, delimiter="\t", quoting=3)
            print train["review"][0]
            return render_template("index.html")
        else:
            print "no success"
          #  return jsonify({'status':'failed'})



class UploadTestingSetAPI(MethodView):

    @staticmethod
    def post(uploadfile):
        file = request.files['file']
        if file and allowed_file(file.filename):
            test = pd.read_csv(file, header=0, delimiter="\t", quoting=3)
            print test["review"][0]
            return render_template("index.html")
        else:
            print "no success"


class StartAnalysisAPI(MethodView):

    @staticmethod
    def get():
        #try:
        test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'testData.tsv'), header=0, delimiter="\t", quoting=3)

        predict_sentiment(test)
        time.sleep(4)
        #return jsonify({'status':'1'})
        #output_csv = open('analysis')
        output_csv = open('analysis.csv')
        # return Response(
        #     output_csv,
        #     mimetype="xml/file",
        #     headers={"Content-disposition":
        #     "attachment; filename=analysis.csv"})

        positive_sentiments = []
        negative_sentiments = []
        with open('Bag_of_Words_model.csv', 'rt') as analysisfile:
            reader = csv.reader(analysisfile)
            headers = next(reader, None)
            # for row in reader:
            #     print row
            #     print len(row)
            for row in reader:
                sentiment = (row[0])[-1]
                each_review = (row[0])[:-1]
                try:
                    if sentiment == '1':
                        positive_sentiments.append(each_review)
                    elif sentiment == '0':
                        negative_sentiments.append(each_review)
                except:
                    print "error"
        response = {}
        response['negative'] = negative_sentiments
        response['positive'] = positive_sentiments

        #return jsonify(response)
        return jsonify(response)


        #except:
         #   return jsonify({'status':'0'})

if __name__ == '__main__':
    try:
        F_DEBUG = os.environ['DEBUG']
    except KeyError:
        F_DEBUG = 'False'

    if F_DEBUG == 'True':
        APP.config.update(DEBUG=True)


    __host_name__ = socket.gethostname()
    __host_or_ip__ = socket.gethostbyname(__host_name__)

    DOWNLOAD_REVIEWS_VIEW = DownloadReviewsAPI.as_view('download_review_api')

    TESTING_API_VIEW = TestingAPI.as_view('testing')

    UPLOAD_TRAINING_VIEW = UploadTrainingSetAPI.as_view('upload_train_api')

    DOWNLOAD_ANALYSIS_VIEW = DownloadAnalysisAPI.as_view('download_csv_api')

    UPLOAD_TESTING_VIEW = UploadTestingSetAPI.as_view('upload_test_api')

    START_ANALYSIS_VIEW = StartAnalysisAPI.as_view('start_analysis_api')

    APP.add_url_rule('/api/1.0/downloadreviewsapi', view_func=DOWNLOAD_REVIEWS_VIEW, methods=['POST'])

    APP.add_url_rule('/api/1.0/testing', view_func=TESTING_API_VIEW, methods=['GET'])

    APP.add_url_rule('/api/1.0/uploadtrainapi', defaults={'uploadfile':None},
                     view_func=UPLOAD_TRAINING_VIEW, methods=['POST'])

    APP.add_url_rule('/api/1.0/uploadtestapi', defaults={'uploadfile':None}, view_func=UPLOAD_TESTING_VIEW, methods=['POST'])

    APP.add_url_rule('/api/1.0/downloadcsvapi', view_func=DOWNLOAD_ANALYSIS_VIEW, methods=['GET'])

    APP.add_url_rule('/api/1.0/startanalysisapi', view_func=START_ANALYSIS_VIEW, methods=['GET'])

    APP.run(threaded=True, host='0.0.0.0', port=8000)
'''
    SERVER_VIEW = ServerAPI.as_view('server_api')
    DATABASE_VIEW = DatabaseAPI.as_view('database_api')
    DATABASE_MEMBER_VIEW = DatabaseMemberAPI.as_view('database_member_api')
    DEPLOYMENT_VIEW = deploymentAPI.as_view('deployment_api')
    DEPLOYMENT_USER_VIEW = deploymentUserAPI.as_view('deployment_user_api')
    XML_VIEW = XMLAPI.as_view("xml_api")
    UPLOAD_CONFIGURATION_VIEW = UploadConfigurationAPI.as_view('uploadconf_api')
    VOLTDBOPERATIONS_VIEW = VoltdbOperations.as_view('voltdboperations_api')
    APP.add_url_rule('/api/1.0/servers/', defaults={'server_id': None},
                     view_func=SERVER_VIEW, methods=['GET'])
    APP.add_url_rule('/api/1.0/servers/<int:database_id>', view_func=SERVER_VIEW, methods=['POST'])
    APP.add_url_rule('/api/1.0/servers/<int:server_id>', view_func=SERVER_VIEW,
                     methods=['GET', 'PUT', 'DELETE'])

    APP.add_url_rule('/api/1.0/databases/', defaults={'database_id': None},
                     view_func=DATABASE_VIEW, methods=['GET'])
    APP.add_url_rule('/api/1.0/databases/<int:database_id>', view_func=DATABASE_VIEW,
                     methods=['GET', 'PUT', 'DELETE'])
    APP.add_url_rule('/api/1.0/databases/', view_func=DATABASE_VIEW, methods=['POST'])
    APP.add_url_rule('/api/1.0/databases/member/<int:database_id>',
                     view_func=DATABASE_MEMBER_VIEW, methods=['GET', 'PUT', 'DELETE'])

    APP.add_url_rule('/api/1.0/deployment/', defaults={'database_id': None},
                     view_func=DEPLOYMENT_VIEW, methods=['GET'])

    APP.add_url_rule('/api/1.0/deployment/<int:database_id>', view_func=DEPLOYMENT_VIEW, methods=['GET', 'PUT'])
    APP.add_url_rule('/api/1.0/deployment/users/<string:username>', view_func=DEPLOYMENT_USER_VIEW,
                     methods=['GET', 'PUT', 'POST', 'DELETE'])
    APP.add_url_rule('/api/1.0/deployment/users/<int:database_id>/<string:username>', view_func=DEPLOYMENT_USER_VIEW,
                     methods=['PUT', 'POST', 'DELETE'])

    APP.add_url_rule('/api/1.0/downloadconfig/', defaults={'xmlfile': None},
                     view_func=XML_VIEW, methods=['GET'])

    APP.add_url_rule('/api/1.0/uploadconfig', defaults={'uploadfile':None},
                     view_func=UPLOAD_CONFIGURATION_VIEW, methods=['POST'])
    APP.add_url_rule('/api/1.0/voltdboperations/', defaults={'database_id':None},
                     view_func=VOLTDBOPERATIONS_VIEW, methods=['GET'])


'''
    #APP.run(threaded=True, host='0.0.0.0', port=8000)
