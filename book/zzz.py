from os import listdir
from os.path import isfile, join
import os
import random
import string
def randomname():
    return str(random.choice(string.letters) + str(random.randint(0,1000)))

onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

for i,f in enumerate(onlyfiles):
    # if (len(f.split(".")) != 2):
    #     print(f)
    #     continue
    # elif ".py" in f:
    #     continue
    # else:
    #     n_name = str(i) + "." + f.split(".")[1]
    #     os.rename(f, n_name)

    if f.split(".")[1] != "jpg":
        print(f)