---
layout: post
title:  "Breadth/depth-first traversal"
categories: notebook
author:
- Jenei Bendegúz
excerpt: Traversing Reddit comment trees
---

<a href="https://colab.research.google.com/github/jben-hun/colab_notebooks/blob/master/breadthDepthCommentTraversal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Implementing breadth- and depth-first traversal and using them to fetch all comments from a reddit comment forest**



PRAW is used to interact with the reddit api, a reddit account and a registered app is required for usage, to get started:

*   <https://praw.readthedocs.io/en/latest/getting_started/quick_start.html>


```python
!pip install -q praw

import praw
from collections import deque

client_id = "" #@param {type:"string"}
client_secret = "" #@param {type:"string"}
user_agent = "" #@param {type:"string"}

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent)
```

    153kB 2.6MB/s 
    204kB 11.5MB/s 


```python
def traverse_comments(comments, *, breadth_first=False):
    queue = deque(comments[:])
    result = []
    while queue:
        e = queue.pop()
        if isinstance(e, praw.models.MoreComments):
            if breadth_first:
                queue.extendleft(e.comments())
            else:
                queue.extend(e.comments())
        else:
            if breadth_first:
                queue.extendleft(e.replies)
            else:
                queue.extend(e.replies)
            result.append(e.body)
    return result
```

**Supply a subreddit url**

Preferably an archived one, so the comments will not change during operation


```python
submission_url = "https://www.reddit.com/r/aww/comments/fo6q11/his_favorite_place_is_his_bed/" #@param {type:"string"}
```

**Our depth-first traversal**


```python
%%time
comments_depthfirst = set(traverse_comments(
    reddit.submission(url=submission_url).comments))
```

    CPU times: user 610 ms, sys: 21.1 ms, total: 631 ms
    Wall time: 1min 10s
    

**Our breadth-first traversal**


```python
%%time
comments_breadthfirst = set(traverse_comments(
    reddit.submission(url=submission_url).comments, breadth_first=True))
```

    CPU times: user 588 ms, sys: 26.4 ms, total: 614 ms
    Wall time: 1min 1s
    

**Built-in breadth-first traversal**


```python
%%time
submission = reddit.submission(url=submission_url)
submission.comments.replace_more(limit=None)
comments_builtin = set([comment.body for comment in submission.comments.list()])
```

    CPU times: user 666 ms, sys: 22.9 ms, total: 688 ms
    Wall time: 1min 7s
    

**Result validation**


```python
print("Results are equivalent:",
      comments_depthfirst ==
      comments_breadthfirst ==
      comments_builtin)
```

    Results are equivalent: True
    

# References

*   <https://praw.readthedocs.io/en/latest/tutorials/comments.html>
*   <https://en.wikipedia.org/wiki/Depth-first_search>
*   <https://en.wikipedia.org/wiki/Breadth-first_search>
