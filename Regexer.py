import re
import os
from pprint import pprint

dict = {}

with os.scandir('./datasets/train-articles') as folder:
    for file in folder:
        dict[re.split("\.", file.name)[0]] = []

with os.scandir('./datasets/train-labels-task1-span-identification') as folder:
    for file in folder:
        key = re.split('\.', file.name)[0]

        with open(file, 'r') as article:
            for line in article.readlines():
                array = line.rstrip().split('\t')
                dict[key].append((array[1], array[2]))

pprint(dict)
