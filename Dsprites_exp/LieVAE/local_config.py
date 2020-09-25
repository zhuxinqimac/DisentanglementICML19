import os
import sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# from config.path import ROOT
from config.path_const import ROOT
from config.parser import dsprites_parser

KEY = 'LieVAE'
RESULT_DIR = ROOT+'{}/'.format(KEY)
ID_STRUCTURE_DICT = {
        'LieVAE' : ('nbatch', 'nconti', 'ncat', 'group_feats_size', 'ntime', 'plamb',
                    'rec', 'spl', 'hes', 'lin', 'ncut', 'dptype', 'piter', 'rseed'),
        }
ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_dsprites_parser():
    parser = dsprites_parser()
    parser.add_argument("--rseed", default = 0, help="random seed", type = int)
    parser.add_argument("--plamb", default = 0.001, help="pairwise cost", type = float)
    parser.add_argument("--group_feats_size", default = 400, help="Group feats size of LieVAE", type = int)
    parser.add_argument("--rec", default = 1, help="Rec hyper of LieVAE", type = float)
    parser.add_argument("--spl", default = 1, help="Split hyper of LieVAE", type = float)
    parser.add_argument("--hes", default = 0, help="Hessian hyper of LieVAE", type = float)
    parser.add_argument("--lin", default = 0, help="Linear hyper of LieVAE", type = float)
    parser.add_argument("--ncut", default = 1, help="Number of cuts of LieVAE", type = int)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a3', help="decay parameter type", type = str)
    parser.add_argument("--nconti", default = 6, help="the dimension of continuous representation", type = int)
    parser.add_argument("--ncat", default = 3, help="size of categorical data", type = int)
    parser.add_argument("--ntime", default = 4, help="When does discrete variable to be learned", type = int)
    parser.add_argument("--niter", default = 300000, help="Number of iters to train", type = int)
    parser.add_argument("--piter", default = 20000, help="Iters as index divider", type = int)
    parser.add_argument("--siter", default = 10000, help="Iters for snapshot", type = int)
    parser.add_argument("--lie_norm_type", default = 'none', help="Lie algebra norm type", type = int)
    return parser

