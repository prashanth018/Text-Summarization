import praw

def getHotPosts(subreddit, posts=25):
    for idx, submission in zip(range(posts),reddit.subreddit(subreddit).hot(limit=posts)):
        if idx == 12:
            print(idx, submission.title)
            # print(idx, submission.comments)
            print(idx, submission.url)
            print(idx, submission.selftext)

reddit = praw.Reddit(client_id='_bmU4jBSzdwwtw',
                     client_secret='TfvvphY2bY6rG5WkbLmZSa5W6Mc',
                     password='test@123',
                     user_agent='testscript by /u/CreepyTelephone',
                     username='CreepyTelephone')


# print(reddit.user.me())
subreddit = "books"
# submission = reddit.subreddit(subreddit).hot(limit=1)
# print(submission.title)
getHotPosts("books")
