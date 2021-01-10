import logging
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
import requests


def crawl(q, data_frame):
    while not q.empty():
        work = q.get()  # fetch new work from the Queue
        try:
            data = requests.get(work[1]).text
            logging.info("Requested..." + work[1])
            data_frame.at[work[0], "GitHubUrl"] = data  # Store data back at correct index
        except Exception as e:
            logging.error('Error with URL check!' + e.__str__())
            data_frame.at[work[0], "GitHubUrl"] = ""
        # signal to the queue that task has been processed
        q.task_done()
    return True


filename = "./data/annotationStore.csv"

df = pd.read_csv(filename)
df = df[(df["Language"] != "Go") & (df["Language"] != "Ruby")]
df = df.drop(["Relevance", "Query", "Notes"], axis=1)

classes = list(df.Language.unique())
df["Language"] = df["Language"].map(lambda x: classes.index(x))

amount = df.count()
num_theads = min(50, df["GitHubUrl"].count())

queue = Queue()
for i, url in df["GitHubUrl"].items():
    # need the index and the url in each queue item.
    modified_url = url.replace('https://github.com/', 'https://raw.githubusercontent.com/')
    modified_url = modified_url.replace('/blob', '')
    queue.put((i, modified_url))

num_theads = min(50, df["GitHubUrl"].count())
for i in range(num_theads):
    logging.debug('Starting thread ', i)
    worker = Thread(target=crawl, args=(queue, df))
    worker.setDaemon(True)  # setting threads as "daemon" allows main program to
    # exit eventually even if these dont finish
    # correctly.
    worker.start()
# now we wait until the queue has been processed
queue.join()

array = df.to_numpy()

class_indices = array[:, 0]
source_codes = array[:, 1]

np.save("../data/classes_of_dataset.npy", np.array(classes))
np.save("../data/class_indices_of_dataset.npy", class_indices)
np.save("../data/source_codes_dataset.npy", source_codes)
