import numpy as np
from tqdm import tqdm
from utils.config import *
from utils.utils_Ent_woz import *

import torch


# fixed random seed
if args['fixed']:
    torch.manual_seed(args['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        torch.backends.cudnn.deterministic = True
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])

early_stop = args['earlyStop']

# Configure models and load data
if args['epoch'] > 0:
    avg_best, cnt, res = 0.0, 0, 0.0
    train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))

