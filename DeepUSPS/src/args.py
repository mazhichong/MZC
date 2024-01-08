import argparse
import sys

# parser for protein generator
def get_args_DeepUSPS():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-l", "--len=",  type=int, required=False, dest='LEN', default=100, 
                        help='sequence length')
    parser.add_argument("-s", "--seq=",  type=str, required=False, dest='SEQ', default="",
                        help='starting sequence')

    parser.add_argument('-o', "--ofas=", type=str, required=False, dest='FAS', default='',
                        help='save final sequence to a FASTA files')
    parser.add_argument("--ocsv=", type=str, required=False, dest='CSV', default='',
                        help='save trajectory to a CSV files')

    parser.add_argument("--SPFESN=",  type=str, required=False, dest='SPDIR', default="../SPSP",
                        help="path to SPFESN network weights")
    parser.add_argument("--RTIDR=", type=str, required=False, dest='RTDIR', default="../RTRT",
                        help="path to RTIDR network weights")

    parser.add_argument('--aa_weight=', type=float, required=False, dest='AA_WEIGHT', default=0.0,
                        help='weight for the aa composition biasing loss term')
    
    args = parser.parse_args()

    return args

