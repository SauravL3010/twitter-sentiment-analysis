import pickle
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import string

# load classifier
def load_trained_classifier(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def load_tweets(tweets_filename):
    tweets = []
    with open(tweets_filename, "r") as tf:
        for line in tf.readlines():
            line = line.rstrip()
            tweets.append(json.loads(line))
    return tweets


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def remove_mentions(input_text):
    return re.sub(r'@\w+', '', input_text)
    
def remove_urls(input_text):
    return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

def remove_punctuation(input_text):
    # Make translation table
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
    return input_text.translate(trantab)

def remove_digits(input_text):
    return re.sub('\d+', '', input_text)

def remove_stopwords(input_text):
    stopwords_list = nltk.corpus.stopwords.words("english")
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 


def tokenize_text(text):
    stopwords = nltk.corpus.stopwords.words("english")
    words = remove_mentions(text)
    words = remove_urls(words)
    words = remove_punctuation(words)
    words = remove_digits(words)
    words = remove_stopwords(words)
    words = nltk.word_tokenize(words)
    words = [w for w in words if w.lower() not in stopwords]
    words = [word.lower() for word in words
            if len(word) > 1 and
            not re.search(r'^\*\*.+$', word) and 
            not re.search(r'^.+\*\*$', word) and                 
            not re.search(r'^[0-9]+', word) and
            not "RT" in word and
            not "\\" in word and 
            not "/" in word and
            word.isalnum()]
    input_text = emoji_pattern.sub(r'', " ".join(words))
    return input_text


def sentiment(tokn_sent):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(tokn_sent)



def tweet_data(tweet):
    '''
    {
        "text": ,
        "tokenized_text": (),
        "pos": ,
        "neg": ,
        "compound" ,
        "urls": [url["display_url"] for url in t[entities][urls]] #
        "hashtags": t[entities][hashtags]
        "len_hashtags": len(t[entities][hashtags])
        "number_of_qustion_marks": text.count("!"),
        "number_of_exclamation_marks": text.count("?"),
        "number_of_user_mentions": len(t[entities][user_mentions])
    }
    '''
    dic = {}
    dic["text"] = tweet["text"]
    dic["tokenized_text"] = tokenize_text(tweet["text"])
    sent = sentiment(dic["tokenized_text"])
    dic["pos"] = sent["pos"]
    dic["neg"] = sent["neg"]
    dic["compound"] = sent["compound"]
    dic["hashtags"] = tweet["entities"]["hashtags"]
    dic["len_hashtags"] = len(dic["hashtags"])
    dic["number_of_qustion_marks"] = tweet["text"].count("?")
    dic["number_of_exclamation_marks"] = tweet["text"].count("!")
    dic["number_of_user_mentions"] = len(tweet["entities"]["user_mentions"])
    return dic

def tweet_features(tweet_data):
    return {
        "pos": tweet_data["pos"],
        "neg": tweet_data["neg"],
        "compound": tweet_data["compound"],
        "len_hashtags": tweet_data["len_hashtags"],
        'number_of_qustion_marks': tweet_data['number_of_qustion_marks'],
        'number_of_exclamation_marks' : tweet_data['number_of_exclamation_marks'],
        'number_of_user_mentions' : tweet_data['number_of_user_mentions']
    }


def return_labels_for_tweets(tweet_filename, classifier):
    tweets = load_tweets(tweet_filename)
    labels = []
    tweet_features_array = []
    for tweet in tweets:
        data = tweet_data(tweet)
        single_tweet_feature = tweet_features(data)
        single_label = classifier.classify(single_tweet_feature)
        labels.append(single_label)
    return labels


def save_labels_to_file(filename, labels):
    with open(filename, 'w') as f:
        for label in labels:
            f.writelines(f'{label}\n')




classifier_filename = "twitter_classifier.pkl"
naive_bayes_classifier = load_trained_classifier(classifier_filename)

tweets_filename = "tweets.txt"
labels_to_output = return_labels_for_tweets(tweets_filename, naive_bayes_classifier)

labels_filename = "labels.txt"
save_labels_to_file(labels_filename, labels_to_output)



################## PLOT STUFF ############################

# plot
import datetime
from matplotlib import pyplot as plt

def get_plot_dict(tweets, labels):
    tweet_dates = []
    for tweet in tweets:
        date = datetime.datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S %z %Y').strftime('%d/%m/%Y')
        tweet_dates.append(date)
    combined_list = list(zip(tweet_dates, labels))
    return combined_list

def get_positives_for_plot(zip_list):
    return [date for (date, label) in zip_list if label=='positive']

def get_all_dates_for_plot(zip_list):
    return [date for (date, label) in zip_list]

    
def get_count_dict(tweet):
    dic = {}
    for date in tweet:
        if date in dic:
            dic[date] += 1
        else:
            dic[date] = 1
    return dic


def convert_for_bar_plot(dic):
    dates = []
    positive_counts = []
    for k, v in dic.items():
        dates.append(k)
        positive_counts.append(v)
    return [dates, positive_counts]


def get_percentages(positives, all_dates):
    dic = {}
    for pos_date in list(positives.keys()):
        if pos_date in all_dates:
            percent = (positives[pos_date] / all_dates[pos_date]) * 100
            dic[pos_date] = [float("%.2f" % percent), positives[pos_date]] 
        else:
            percent = 100
            dic[pos_date] = [percent, positives[pos_date]]
    return dic

def make_list_for_plot(lst):
    dates = []
    percentages = []
    positives = []
    for (k, v) in lst.items():
        dates.append(k)
        percentages.append(v[0])
        positives.append(v[1])
    return {"dates": dates, "positives": positives, "percentages": percentages}


def final_plot(plot_list, plot_filename):  
    plt.figure(figsize=(20, 10))
    graph = plt.bar(plot_list["dates"], plot_list["positives"], color="black", align='center', width=0.5)
    plt.title('Plot of daily percentages of vaccine-hesitant (positive) tweets')
    plt.xticks(rotation=90)
    plt.ylabel("Number of vaccine-hesitant tweets on a daily basis")

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                 y+height*1.01,
                 str(plot_list["percentages"][i])+'%',
                 ha='center',
                 weight='light')
        i+=1
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.show()


plot_tweets = load_tweets(tweets_filename)
plot_labels = labels_to_output
all_tweets_for_plot = get_plot_dict(plot_tweets, plot_labels)
positive_dates_only = get_positives_for_plot(all_tweets_for_plot)
all_dates = get_all_dates_for_plot(all_tweets_for_plot)
date_percentages = get_percentages(get_count_dict(positive_dates_only), get_count_dict(all_dates))
plot_list = make_list_for_plot(date_percentages)
plot_filename = "plot.png"
final_plot(plot_list, plot_filename)




