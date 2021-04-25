---
layout: post
title:  "Markov sentences"
categories: notebook
author:
- Jenei Bendegz
excerpt: Generate reddit comments with a markov chain
---

<a href="https://colab.research.google.com/github/jben-hun/colab_notebooks/blob/master/markovSentences.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Text generation with nth order Markov chain models trained on reddit data**

The probability $P(\ word_{m}\ \mid\ word_{m-1}\ \land\ word_{m-2}\ \land\ \cdots\ \land\ word_{m-n}\ )$, where $n$ is the order of the model, is proportional to the relative occurrences of such sequence of words in the training dataset.

Example chain with a model of order $n=2$:

$\cdots\ word_{m-3}\ (word_{m-2}\ word_{m-1})\rightarrow(word_{m})\ word_{m+1}\ \cdots$

# Implementation

PRAW is used to interact with the reddit api, a reddit account and a registered app is required for usage, to get started:

*   <https://praw.readthedocs.io/en/latest/getting_started/quick_start.html>


```python
!pip install -q praw

import praw
import re
import random
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import deque

pd.set_option("max_colwidth", None)

client_id = "" #@param {type:"string"}
client_secret = "" #@param {type:"string"}
user_agent = "" #@param {type:"string"}

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent)
```

    153kB 2.7MB/s 
    204kB 5.9MB/s 


```python
SUBREDDITS = ("askreddit", "explainlikeimfive", "dankmemes")


class RedditMarkovChain:
    def __init__(
            self,
            subreddit,
            order=1,
            sentence_limit=1000,
            begin_str="*BEGIN*",
            end_str="*END*",
            cycle_str="*CYCLE*",
            train_split=(0.9),
            test_split=None):
        self.__subreddit = subreddit
        self.__order = order
        # first order: word1 -> word2
        # second order: (word1, word2) -> word3
        # ...
        self.__sentence_limit = sentence_limit
        self.__begin_str = begin_str
        self.__end_str = end_str
        self.__cycle_str = cycle_str
        self.__train_split = train_split

        self.__test_split = (test_split if test_split is not None
                            else (1.0 - train_split))
        
        self.__sentences = self.mine_subreddit(
            subreddit=reddit.subreddit(self.subreddit),
            sentence_limit=self.sentence_limit)

        self.model = self.__build_model()

    def __build_model(self):
        """Build a markov chain model from extracted sentences"""

        model = defaultdict(lambda: defaultdict(lambda: 0))

        for sentence in self.train_sentences:
            words = ([self.begin_str] +
                     self.__split_sentence(sentence) +
                     [self.end_str])
            for i in range(1, len(words)):
                for j in range(i-1, i-self.order-1, -1):
                    if j < 0:
                        break
                    key = self.__convert_to_key(words[j:i])
                    model[key][words[i]] += 1

        return model

    def __convert_to_key(self, values):
        """Pads with None values to until the length is equal to the order"""
        assert len(values) <= self.order
        return tuple([None]*(self.order - len(values)) + values)

    def __get_longest_key(self, l, keys):
        """Returns the longest subkey that exist in the model,
        or the full key if no subkey was found"""

        for i in range(len(l)-1):
            key = tuple(l[i:])
            if key in keys:
                return key

        # if no existing key was found, return the full key, so the defaultdict
        # creates an entry for it with a count value of zero
        return tuple(l)

    def __split_sentence(self, sentence):
        """Split sentences into words"""
        return re.findall(r"((?:[\w']+)|(?:[,!.?]))", sentence)

    def __get_prob(self, key, word):
        """Get single probability from word counts"""
        return (0 if word not in self.model[key]
                else self.model[key][word]/sum(self.model[key].values()))

    def __get_all_probs(self, key):
        """Get all probabilities from word counts"""
        n = sum(self.model[key].values())
        return [v/n for v in self.model[key].values()]

    def generate(self, method="sample"):
        """Generate text using the created markov chain model

        method:
            expected: choose most likely words, infinite cycles are possible
            random: choose words uniformly
            sample: choose words based on the modeled probabilities
        """

        sentence = []
        word = self.begin_str
        key = self.__convert_to_key([word])

        if method == "expected":
            used = set()

        while True:
            if method == "expected":
                word = max(
                    self.model[key].items(), key=lambda x: x[1])[0]
            elif method == "random":
                word = random.choice(tuple(self.model[key].items()))[0]
            elif method == "sample":
                words = tuple(self.model[key].keys())
                probs = self.__get_all_probs(key)
                word = np.random.choice(words, p=probs)
            if word == self.end_str:
                break

            sentence.append(word)

            key = self.__convert_to_key(sentence[-self.order:])

            if method == "expected":
                if key in used:
                    sentence.append(f"{self.cycle_str}")
                    break
                used.add(key)

        return (" ".join(sentence).replace(" .", ".")
                                  .replace(" ?", "?")
                                  .replace(" !", "!")
                                  .replace(" ,", ","))

    def classify(self, sentence):
        """Deduce the most likely source of a sentence"""

        p = 1

        words = ([self.begin_str] +
                 self.__split_sentence(sentence) +
                 [self.end_str])

        for i in range(1, len(words)):
            key = self.__get_longest_key(
                self.__convert_to_key(words[max(0, i-self.order):i]),
                self.model.keys())
            p *= self.__get_prob(key, words[i])

        return p

    @staticmethod
    def mine_subreddit(subreddit, sentence_limit):
        """Extract clean sentences from submissions and comments"""

        # re that matches clean sentences
        matcher = re.compile(r"(?:[.!?] |^)[A-Z][\w', ]+[.!?](?= [A-Z]|$)")

        sentences = []

        with tqdm.tqdm(total=sentence_limit) as pbar:
            for submission in subreddit.hot(limit=None):
                sentences += matcher.findall(submission.title)
                sentences += matcher.findall(submission.selftext)

                submission.comment_sort = "best"

                comments = [
                    comment.body for comment in submission.comments.list()
                    if not isinstance(comment, praw.models.MoreComments)]

                for comment in comments:
                    sentences += matcher.findall(comment)

                if len(sentences) >= sentence_limit:
                    random.shuffle(sentences)
                    pbar.update(sentence_limit - pbar.n)
                    break
                else:
                    pbar.update(len(sentences) - pbar.n)

        return [sentence.lstrip(".!? ")
                        .replace("won't", "will not")
                        .replace("n't", " not")
                        .replace("'m", " am")
                        .replace("'re", " are")
                        for sentence in sentences[:sentence_limit]]

    @property
    def subreddit(self):
        return self.__subreddit

    @property
    def sentences(self):
        return self.__sentences

    @property
    def order(self):
        return self.__order

    @property
    def sentence_limit(self):
        return self.__sentence_limit

    @property
    def begin_str(self):
        return self.__begin_str

    @property
    def end_str(self):
        return self.__end_str

    @property
    def cycle_str(self):
        return self.__cycle_str

    @property
    def train_split(self):
        return self.__train_split

    @property
    def test_split(self):
        return self.__test_split

    @property
    def train_sentences(self):
        return self.sentences[:int(len(self.sentences)*self.train_split)]

    @property
    def test_sentences(self):
        return self.sentences[int(len(self.sentences)*self.train_split):]
```

# Demo

**Construct markov chains for each specified subreddit**


```python
chains = {subreddit: RedditMarkovChain(subreddit, order=2, sentence_limit=1000)
          for subreddit in SUBREDDITS}
```

    100% 1000/1000 [00:14<00:00, 68.29it/s]
    100% 1000/1000 [00:14<00:00, 67.01it/s]
    100% 1000/1000 [00:37<00:00, 26.85it/s]
    

**Deriving most probable sentence for each model**


```python
for subreddit, chain in chains.items():
    print(f"{subreddit}: {chain.generate('expected')}")
```

    askreddit: I was at a young age.
    explainlikeimfive: The problem with strictly widening a base, while maintaining perpendicular wheels, is there really any point?
    dankmemes: I am not a arest' of the movie.
    

**Generating new text**


```python
dict_data = defaultdict(lambda: [])

for subreddit, chain in chains.items():
    for _ in range(5):
        sentence = chain.generate()
        dict_data["sentence"].append(sentence)
        dict_data["model"].append(subreddit)

        for k, v in chains.items():
            p = v.classify(sentence)
            dict_data[f"P({k})"].append(p)

display(pd.DataFrame(dict_data))
```


<div style="overflow: auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence</th>
      <th>model</th>
      <th>P(askreddit)</th>
      <th>P(explainlikeimfive)</th>
      <th>P(dankmemes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>And it was great to see us do better in the neck.</td>
      <td>askreddit</td>
      <td>3.283425e-07</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yep.</td>
      <td>askreddit</td>
      <td>3.333333e-03</td>
      <td>0.000000e+00</td>
      <td>0.002222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>They have the Democrats would control the Senate, and it would cost the city less money to simply buy Norman and his mum a house far far away than to keep his mouth shut about the actual state of being in love.</td>
      <td>askreddit</td>
      <td>6.977041e-11</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mitch's interests are focused on my own spit I want that put in my own car.</td>
      <td>askreddit</td>
      <td>1.102293e-06</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Congrats, onward with your journey.</td>
      <td>askreddit</td>
      <td>5.555556e-04</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A great example of this lap the runners will be separated by the stagger so there are parts of the lower body are some of these things.</td>
      <td>explainlikeimfive</td>
      <td>0.000000e+00</td>
      <td>7.419278e-10</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Thanks for powerfully refuting your own horrible idea.</td>
      <td>explainlikeimfive</td>
      <td>0.000000e+00</td>
      <td>1.111111e-03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The answer is not really explain it well, but also a matter of how anything floats is that you can try this easily.</td>
      <td>explainlikeimfive</td>
      <td>0.000000e+00</td>
      <td>1.627817e-11</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PC games are made to run properly on your system, kidneys and digestion.</td>
      <td>explainlikeimfive</td>
      <td>0.000000e+00</td>
      <td>2.314815e-05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enough slack that it will interfere with the clock and yourself.</td>
      <td>explainlikeimfive</td>
      <td>0.000000e+00</td>
      <td>1.736111e-06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Im a minor and seeing those girls in those provocative positions and clothing isnt ok.</td>
      <td>dankmemes</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.001111</td>
    </tr>
    <tr>
      <th>11</th>
      <td>What I said, with utter confidence, that it'd become a square.</td>
      <td>dankmemes</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000139</td>
    </tr>
    <tr>
      <th>12</th>
      <td>How.</td>
      <td>dankmemes</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.001111</td>
    </tr>
    <tr>
      <th>13</th>
      <td>The cum accelerates.</td>
      <td>dankmemes</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.005556</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Well played.</td>
      <td>dankmemes</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.001111</td>
    </tr>
  </tbody>
</table>
</div>


**Classifying real text**

Classifying isn't exactly ideal this way, as probabilities turn zero when an unprecedented word connection is met. Check out multinomial naive Bayes for a simple text classifier.


```python
dict_data = defaultdict(lambda: [])

for subreddit, chain in chains.items():
    for sentence in chain.test_sentences[:5]:
        dict_data["sentence"].append(sentence)
        dict_data["source"].append(subreddit)

        for k, v in chains.items():
            p = v.classify(sentence)
            dict_data[f"P({k})"].append(p)

display(pd.DataFrame(dict_data))
```


<div style="overflow: auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence</th>
      <th>source</th>
      <th>P(askreddit)</th>
      <th>P(explainlikeimfive)</th>
      <th>P(dankmemes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sixth grade end of year trip, we were at a ropes course park where you'd rock climb and walked across tall catwalks and the like.</td>
      <td>askreddit</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Our reality is a fiction created by a higher civilization of higher beings who wrote a story and transformed it into a reality and every move, breath, even blink, is already programmed.</td>
      <td>askreddit</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mitch is really old.</td>
      <td>askreddit</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Choking to death alone is honestly my biggest fear.</td>
      <td>askreddit</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The logic was similar for us, yes.</td>
      <td>askreddit</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>On my PC, just the BIOS takes longer than that.</td>
      <td>explainlikeimfive</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>No, only 4 left and 5 right.</td>
      <td>explainlikeimfive</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HAHA.</td>
      <td>explainlikeimfive</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Why is it that being slapped in the face will make you cry, stubing your toe makes you inhale rendering you speechless and being burnt by hot water will make you growl or scream?</td>
      <td>explainlikeimfive</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kernel API is for abstracting away the nasty parts of communicating with the operating system.</td>
      <td>explainlikeimfive</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>My pain is far greater than yours!</td>
      <td>dankmemes</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Really?</td>
      <td>dankmemes</td>
      <td>0.001111</td>
      <td>0.0</td>
      <td>0.002222</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Weird flex warning!</td>
      <td>dankmemes</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Is this 1984?</td>
      <td>dankmemes</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>After ten spurts you start to worry.</td>
      <td>dankmemes</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


# References

*   <https://en.wikipedia.org/wiki/Markov_chain>
*   <https://www.reddit.com/r/SubredditSimulator/comments/3g9ioz/what_is_rsubredditsimulator/>
*   <https://www.reddit.com/r/SubSimulatorGPT2/comments/btfhks/what_is_rsubsimulatorgpt2/>
