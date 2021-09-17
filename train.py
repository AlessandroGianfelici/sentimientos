from sentimientos import Model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, MAX_N_WEMBS
import pandas as pd
import os

url = r'https://drive.google.com/u/0/uc?id=1Vy2pu3wx-7EkNlvDLIUSUeu5NWnuSOGu&export=download'

review_data = pd.read_csv(os.path.join("sentimientos", "train_data", "reviews", "raw_data.txt"))
wikipedia_data = pd.read_csv(url)
