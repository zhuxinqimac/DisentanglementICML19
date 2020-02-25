import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from model import Model

from config.path import subdirs5resultdir, muldir2mulsubdir

from utils.datasetmanager import dsprites_manager, shapes_3d_manager
from utils.format_op import FileIdManager

from local_config import local_dsprites_parser, ID_STRUCTURE, KEY

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    parser = local_dsprites_parser()
    parser.add_argument("--n_pairs", default = 10000, help="number of image pairs to generate", type = int)
    parser.add_argument("--pair_batch_size", default = 100, help="batch size for pair generation", type = int)
    parser.add_argument("--latent_type", default = 'onedim', help="latent type", type = str, choices=['onedim', 'fulldim'])
    parser.add_argument("--pair_include_discrete", action='store_true', help='include discrete in image pair generation')
    args = parser.parse_args() # parameter required for model

    ROOT_DIRS = {
            'dsprites': '/share2/xqzhu/repo_results/DisentanglementICML19/Results_dsprites/',
            '3dshapes': '/share2/xqzhu/repo_results/DisentanglementICML19/Results_3dshapes/'
            }
    ROOT = ROOT_DIRS[args.dataset]
    RESULT_DIR = ROOT+'{}/'.format(KEY)

    fim = FileIdManager(ID_STRUCTURE)

    np.random.seed(args.rseed)
    FILE_ID = fim.get_id_from_args(args)
    SAVE_DIR, LOG_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, False)
    SAVE_SUBDIR, ASSET_SUBDIR = muldir2mulsubdir([SAVE_DIR, ASSET_DIR], FILE_ID, False)

    if args.dataset == 'dsprites':
        dm = dsprites_manager()
    else:
        dm = shapes_3d_manager()
    dm.print_shape()

    model = Model(dm, LOG_DIR+FILE_ID+'.log', args)
    model.initialize()
    model.restore(save_dir=SAVE_SUBDIR)
    model.generate_image_pairs(batch_size=args.pair_batch_size, asset_dir=ASSET_SUBDIR, 
            n_pairs=args.n_pairs, include_discrete=args.pair_include_discrete)
