'''
import requests
import requests.auth
client_auth = requests.auth.HTTPBasicAuth('_bmU4jBSzdwwtw', 'TfvvphY2bY6rG5WkbLmZSa5W6Mc')
post_data = {"grant_type": "password", "username": "CreepyTelephone", "password": "test@123"}
headers = {"User-Agent": "ChangeMeClient/0.1 by CreepyTelephone"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
print response.json()
headers = {"Authorization": "bearer "+response.json()['access_token'], "User-Agent": "ChangeMeClient/0.1 by CreepyTelephone"}
response = requests.get("https://oauth.reddit.com/api/v1/me", headers=headers)
print response.json()
'''

import praw
reddit = praw.Reddit(client_id='_bmU4jBSzdwwtw',
                     client_secret='TfvvphY2bY6rG5WkbLmZSa5W6Mc',
                     password='test@123',
                     user_agent='testscript by /u/CreepyTelephone',
                     username='CreepyTelephone')

print reddit.user.me()
