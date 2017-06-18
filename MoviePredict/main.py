## ENVIRIONMENT IS ACTIVATED AT THIS POINT - BASICALLY THE REQUIREMENTS ARE ALREADY INSTALLED (SEE REQUIREMENTS FILE)
def pretty_print_review_and_label(i):  ####### THIS is where we merge the labels and reviews 
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print("number of reviews are ")
print(len(reviews))

print("EXAMPLE OF A REVIEW IS ")
print(reviews[0])

print("THE ASSOCIATED RATING IS ")
print(labels[0])
print('\n')


## print out some examples, in order to create a predictive theory lets see if we can figure it out by eye first
print('## print out some examples, in order to create a predictive theory lets see if we can figure it out by eye first \n')
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)


#lets count up all the positive and negative words 


from collections import Counter
import numpy as np


# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()


# Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i in range(len(reviews)): # len(reviews) gives 25000 paragraphs
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):  #each paragraph is split into words and each word passed to word var
            positive_counts[word] += 1    #all these words are in positively rated paragraph
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1


# Examine the counts of the most common words in positive reviews
print('# Examine the counts of the most common words in positive reviews')
positive_counts.most_common()


# Examine the counts of the most common words in negative reviews
print('# Examine the counts of the most common words in negative reviews')
negative_counts.most_common()


# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times
    
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio


print('lets look at the rations of ratings now')

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))	


print('Convert ratios to logs')
for word, ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[word] = np.log(ratio)

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))



