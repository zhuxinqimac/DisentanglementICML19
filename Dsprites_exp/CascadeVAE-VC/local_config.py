import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from config.path import ROOT
from config.parser import dsprites_parser

KEY = 'CascadeVAE-VC'
RESULT_DIR = ROOT+'{}/'.format(KEY)
ID_STRUCTURE_DICT = {
        'CascadeVAE-VC' : ('nbatch', 'nconti', 'ncat', 'ntime', 'plamb', 
            'beta_min', 'beta_max', 'dptype', 'C_lambda', 'F_beta', 'disc_type', 
                           'delta_std', 'switch_dis', 'rseed'),
        }
ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_dsprites_parser():
    parser = dsprites_parser()
    parser.add_argument("--rseed", default = 0, help="random seed", type = int)
    parser.add_argument("--plamb", default = 0.001, help="pairwise cost", type = float)
    parser.add_argument("--beta_min", default = 0.1, help="min value for +beta*kl_cost", type = float)
    parser.add_argument("--beta_max", default = 10.0, help="max value for +beta*kl_cost", type = float)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a3', help="decay parameter type", type = str)
    parser.add_argument("--nconti", default = 6, help="the dimension of continuous representation", type = int)
    parser.add_argument("--ncat", default = 3, help="size of categorical data", type = int)
    parser.add_argument("--ntime", default = 4, help="When does discrete variable to be learned", type = int)
    parser.add_argument("--C_lambda", default = 1, help="Continuous lambda for VC model", type = float)
    parser.add_argument("--F_beta", default = 0, help="Beta of feature map loss for VC model", type = float)
    parser.add_argument("--disc_type", default = 'normal', help="Disc type", type = str, choices=['normal', 'simple'])
    parser.add_argument("--delta_std", default = 2, help="Delta std for discriminator", type = float)
    parser.add_argument("--switch_dis", action='store_true', help="If switch discrete")
    return parser

