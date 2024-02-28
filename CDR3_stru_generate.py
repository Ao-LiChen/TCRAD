import os
from os.path import join, dirname
import argparse
import numpy as np

import paddle

from utils.utils_stru import tree_map
from utils.model_paddle_bert import Run_paddleModel
from alphafold_paddle.model import features
from alphafold_paddle.model import config
from alphafold_paddle.model import utils
from alphafold_paddle.common import protein, residue_constants
from alphafold_paddle.data.data_utils import single_sequence_to_features
import pickle
import copy

with open('AA.pkl', 'rb') as file:
    AA = pickle.load(file)

def sequence_to_batch(cdr3,peptide,BindingMode,model_config):
    """
    make batch data with single sequence
    """
    raw_features = single_sequence_to_features(cdr3, '>test')
    feat = features.np_example_to_features(np_example=raw_features, config=model_config)
    aatypes = copy.deepcopy(feat['aatype'][0, :]).tolist()

    l = len(aatypes)
    left = (20 - l) // 2
    right = 20 - l - left
    seq = [22] * left + aatypes + [22] * right + [23]

    if BindingMode:
        seg = [0] * 16 + [1] * 21
        seg = np.array(seg).reshape(1, -1)

        peptide_ = [AA[i] for i in peptide] + [22] * (15 - len(peptide)) + [23]
        seq = peptide_ + seq

        seq = np.array(seq).reshape(1, -1)

        batch = {
            "name": cdr3+"_"+peptide,
            "feat": tree_map(lambda v: paddle.to_tensor(v[None, ...]), feat),
            "bounded": True,
            "peptide": peptide,
            "seq": seq,
            "seg": seg,
            "label": {},
        }
    else:
        seg = [0] * 21
        seg = np.array(seg).reshape(1, -1)
        seq = np.array(seq).reshape(1, -1)

        batch = {
            "name": cdr3,
            "feat": tree_map(lambda v: paddle.to_tensor(v[None, ...]), feat),
            "bounded": False,
            "seq": seq,
            "seg": seg,
            "label": {},
        }
    return batch

def postprocess(batch, results, output_dir):
    """save unrelaxed pdb"""
    batch['feat'] = tree_map(lambda x: x[0].numpy(), batch['feat'])     # slice the 1st item
    results = tree_map(lambda x: x[0].numpy(), results)

    results.update(utils.get_confidence_metrics(results))
    plddt = results['plddt']
    plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(batch['feat'], results, b_factors=plddt_b_factors)
    pdb_str = protein.to_pdb(prot)

    with open(join(output_dir, batch['name'] + '.pdb'), 'w') as f:
        f.write(pdb_str)


def main(args):
    """main function"""
    ### create model
    af2_model_name="seq21_pair48_l8_512_vio2"
    af2_model_config = config.model_config(af2_model_name)

    af2_model_config.data.eval.delete_msa_block = False

    batch=sequence_to_batch(args.CDR3,args.Peptide,args.BindingMode,af2_model_config)

    model = Run_paddleModel(af2_model_config)

    if args.BindingMode:
        model.load_params("./model/Bounded_model.pdparams")
    else:
        model.load_params("./model/Unbounded_model.pdparams")

    ### make predictions
    model.eval()

    batch['feat'] = tree_map(lambda v: paddle.to_tensor(v), batch['feat'])
    with paddle.no_grad():
        results = model(batch, bounded=batch['bounded'])


    output_dir = "./output_stru"
    os.makedirs(output_dir, exist_ok=True)
    postprocess(batch, results, output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--CDR3", type=str, help='CDR3 Sequence',default="CATSALGDTQYF")
    parser.add_argument("--Peptide", type=str, help='Peptide Sequence')
    parser.add_argument("--BindingMode", type=bool, help='Binding or Unbinding')
    args = parser.parse_args()

    if args.Peptide is None:
        args.BindingMode = False
    else:
        args.BindingMode = True

    main(args)
