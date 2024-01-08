import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, script_dir+'/src/')

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import *
from src.wra import *
from src.RTIDR import *
from src.args import get_args_DeepUSPS
from time import time

def main():
    ########################################################
    # 0. process inputs
    ########################################################

    args = get_args_DeepUSPS()

    aa_valid = np.arange(20)

    # initialize starting sequence
    if args.SEQ != "":
        L = len(args.SEQ)
        seq0 = args.SEQ
    else:
        L = args.LEN
        seq0 = idx2aa(np.random.choice(aa_valid, L))

    ########################################################
    # 1. generate RTIDR distributions
    ########################################################
    print("Generating RTIDR distributions...")
    rtidr = get_RTIDR(L,args.RTDIR)


    ########################################################
    # 2. run WRA
    ########################################################
    print('Generating SPFESN distributions...')
    traj,seq = wra(args.SPDIR,seq0,rtidr,args.AA_WEIGHT)

    ########################################################
    # 3. save results
    ########################################################
    if args.CSV != "":
        df = pd.DataFrame(traj, columns = ['step', 'sequence', 'score'])
        df.to_csv(args.CSV, index = None)

    if args.FAS != "":
        with open(args.FAS,'w') as f:
            f.write(">seq\n%s\n"%(seq))

if __name__ == '__main__':
    s = time()
    main()
    print(f'time:{time() - s:.2f}s')
