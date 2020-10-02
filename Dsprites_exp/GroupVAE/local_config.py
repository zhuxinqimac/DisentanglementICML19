import os
import sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# from config.path import ROOT
from config.path_const import ROOT
from config.parser import dsprites_parser

KEY = 'GroupVAE'
RESULT_DIR = ROOT+'{}/'.format(KEY)
ID_STRUCTURE_DICT = {
        'GroupVAE' : ('nbatch', 'nconti', 'ncat', 'group_feats_size', 'ntime', 'plamb', 'beta',
                      'gmat', 'spl', 'ncut', 'hes', 'lin', 'hessian_type', 'n_act_points', 'piter', 'rseed'),
        }
ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_dsprites_parser():
    parser = dsprites_parser()
    parser.add_argument("--rseed", default = 0, help="random seed", type = int)
    parser.add_argument("--plamb", default = 0.001, help="pairwise cost", type = float)
    parser.add_argument("--group_feats_size", default = 400, help="Group feats size of GroupVAE", type = int)
    parser.add_argument("--n_act_points", default = 10, help="n_act_points in GroupVAE", type = int)
    parser.add_argument("--hessian_type", default = 'no_act_points', help="hessian type if using act_points", type = str)
    parser.add_argument("--beta", default = 1, help="Beta hyper of GroupVAE", type = float)
    parser.add_argument("--gmat", default = 0, help="Group mat mul hyper of GroupVAE", type = float)
    parser.add_argument("--hes", default = 0, help="Hessian hyper of GroupVAE", type = float)
    parser.add_argument("--lin", default = 0, help="Linear hyper of GroupVAE", type = float)
    parser.add_argument("--spl", default = 0, help="Split hyper of GroupVAE", type = float)
    parser.add_argument("--ncut", default = 0, help="Number of split cut hyper of GroupVAE", type = int)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a3', help="decay parameter type", type = str)
    parser.add_argument("--nconti", default = 6, help="the dimension of continuous representation", type = int)
    parser.add_argument("--ncat", default = 3, help="size of categorical data", type = int)
    parser.add_argument("--ntime", default = 4, help="When does discrete variable to be learned", type = int)
    parser.add_argument("--niter", default = 300000, help="Number of iters to train", type = int)
    parser.add_argument("--piter", default = 20000, help="Iters as index divider", type = int)
    parser.add_argument("--siter", default = 10000, help="Iters for snapshot", type = int)
    return parser

