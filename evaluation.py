#Use your sentiment classifier to evaluate the reviews in the friendships.reviews.txt file, which can be
#found in the Moodle resource folder. (Not all people in this file have given reviews!). Record the scores;
#It may be helpful to treat the scores of 1-2 as negative or 4-5 as positive! It may (or may not) be helpful
#to remember that there is one community where scores are particularly positive, because Amazon has
#run a very convincing mass-advertising campaign in that particular community.

from loguru import logger
from communities import create_communities
from sentiment import naive_bayes
from sklearn.metrics import precision_recall_fscore_support

score_classes = {
    '0': None,
    '1': 0,
    '2': 0,
    '3': None,
    '4': 1,
    '5': 1,
}
number_of_communities = 4

class User:
    def __init__(self, name, friends, review):
        self.name = name
        self.friends = friends
        self.review = review
        self.score = 0

class Result:
    def __init__(self, name, cluster, purchase, score):
        self.name = name
        self.cluster = int(cluster)
        self.would_purchase = purchase
        self.score = score


def load_data(filename):
    users_with_reviews = []
    users_without_reviews = []

    with open(filename, "r") as file:
        for line in file.readlines():
            split = [x.strip() for x in line.split(':')]

            if split[0] == 'user':
                name = split[1]
            elif split[0] == 'friends':
                friends = split[1].split('\t')
            elif split[0] == 'review':
                review = split[1]
                if review == '*':
                    users_without_reviews.append(User(name,friends,review))
                else:
                    users_with_reviews.append(User(name,friends,review))
    return users_without_reviews, users_with_reviews

def load_results(filename):
    results = []

    with open(filename, "r") as file:
        for line in file.readlines():
            split = [x.strip() for x in line.split(':')]

            if split[0] == 'user':
                name = split[1]
            if split[0] == 'cluster':
                cluster = split[1]
            if split[0] == 'score':
                score = score_classes[split[1]]
            elif split[0] == 'purchase':
                purchase = 1 if split[1] == 'yes' else 0
                results.append(Result(name,cluster,purchase,score))
            
    return results

def will_they_buy(community, users_without_reviews, users_with_reviews):
    model = naive_bayes()
    for user in users_with_reviews:
        user.score = model.predict(user.review)
    logger.info('Computed predictions')

    for user in users_without_reviews:
        user.score = infer_score_from_friends(user, users_with_reviews, community)
    logger.info('Infered predictions')

    return users_without_reviews, users_with_reviews


def infer_score_from_friends(user, users_with_reviews, community):
    sum_score = 0

    # All the user's friends that have given a review.
    friends_with_reviews = [friend for friend in users_with_reviews if friend.name in user.friends]
    num_friends_with_reviews = len(friends_with_reviews)

    # Determines the community of the user
    user_community = [com for com, names in community.items() if user.name in names]

    for friend in friends_with_reviews:
        friend_score = friend.score
        
        # If the friend is from another community the score counts 10 times as much
        # If the friend is from Kyle the score counts 10 times as much
        friend_community = [com for com, names in community.items() if friend.name in names]
        if(friend.name == 'kyle' or user_community != friend_community):
            num_friends_with_reviews += 10
            friend_score *= 10
        sum_score += friend_score 
    
    # Score is the average score of all friends scores
    score = sum_score if num_friends_with_reviews == 0 else round(sum_score / num_friends_with_reviews)
    return score

def compute_measures(actual, users_without_reviews, users_with_reviews, community, scipy_community):
    ''' 
    To compute accuracy measures we still differentiate between users who 
    gave a review a review and those who didn't. For those who did we compare
    on the score given. For those who did not we compare on wheter they 
    purchased the product or not. 
    '''

    # Sort the lists according to the name of the user, s.t. all lists
    # are in the same order.
    actual.sort(key=lambda x: x.name)
    users_with_reviews.sort(key=lambda x: x.name)
    users_without_reviews.sort(key=lambda x: x.name)
    
    compute_review_scores(actual, users_with_reviews)
    compute_review_purchaces(actual, users_without_reviews)
    compute_cluster_accuracy(actual, community, scipy_community)


def compute_review_scores(actual, users_with_reviews):
    results = [act for act in actual if act.score != None]
    user_scores = [user.score for user in users_with_reviews if user.name in [act.name for act in results]]
    results = [act.score for act in results]

    accuracy = len([1 for pred, res in zip(user_scores, results) if pred == res]) / len(results)
    precision, recall, fscore, support = precision_recall_fscore_support(results, user_scores)
    write_results_to_file("SCORE ACCURACY", "Accuracy: " + str(accuracy), "Precision: " + str(precision), "Recall: " +  str(recall), "Fscore: " +  str(fscore), "Support: " + str(support))


def compute_review_purchaces(actual, users_without_reviews):
    results = [act.would_purchase for act in actual if act.name in [user.name for user in users_without_reviews]]
    user_scores = [user.score for user in users_without_reviews]

    accuracy = len([1 for pred, res in zip(user_scores, results) if pred == res]) / len(results)
    precision, recall, fscore, support = precision_recall_fscore_support(results, user_scores)

    write_results_to_file("PURCHASE ACCURACY", "Accuracy: " + str(accuracy), "Precision: " + str(precision), "Recall: " +  str(recall), "Fscore: " +  str(fscore), "Support: " + str(support))


def compute_cluster_accuracy(actual, community, scipy_community):
    my_communities = f"My communities 1: {len(community[0])}, 2: {len(community[1])}, 3: {len(community[2])}, 4: {len(community[3])}, 5: {len(community[4])}, 6: {len(community[5])}, 7: {len(community[6])}, 8: {len(community[7])}, 9: {len(community[8])}, 10: {len(community[8])}"
    scipy_communities = f"My communities 1: {len(scipy_community[0])}, 2: {len(scipy_community[1])}, 3: {len(scipy_community[2])}, 4: {len(scipy_community[3])}, 5: {len(scipy_community[4])}, 6: {len(scipy_community[5])}, 7: {len(scipy_community[6])}, 8: {len(scipy_community[7])}, 9: {len(scipy_community[8])}, 10: {len(scipy_community[9])}"
    actual_communities = f"Actual community 1: {sum(map(lambda x : x.cluster == 1, actual))}, 2: {sum(map(lambda x : x.cluster == 2, actual))}, 3: {sum(map(lambda x : x.cluster == 3, actual))}, 4: {sum(map(lambda x : x.cluster == 4, actual))} "
    write_results_to_file("COMMUNITY ACCURACY", my_communities, scipy_communities, actual_communities)


def write_results_to_file(*args):
    with open('results2.txt', "a") as file:
        for arg in args:
            file.write(arg + '\n')
        file.write('\n')


if __name__ == "__main__":
    users_without_reviews, users_with_reviews = load_data('data/friendships.reviews.txt')
    actual = load_results('data/friendships.reviews.results.txt')
    logger.info('Loaded friendships data')

    community = create_communities(number_of_communities, type='spectral')
    community_scipy = create_communities(number_of_communities, type='scipy')
    logger.info('Created communities')
    users_without_reviews, users_with_reviews = will_they_buy(community,users_without_reviews, users_with_reviews)

    compute_measures(actual, users_without_reviews, users_with_reviews,  community, community_scipy)

