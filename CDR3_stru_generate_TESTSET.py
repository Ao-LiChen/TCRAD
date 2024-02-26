import os
from os.path import join
import argparse
import numpy as np
import paddle
from utils.utils_stru import get_model_parameter_size, tree_map
from utils.model_paddle_bert import Run_paddleModel
from alphafold_paddle.model import config
from alphafold_paddle.model import utils
from alphafold_paddle.common import protein, residue_constants
import pickle
import copy

def read_fasta_file(fasta_file):
    """
    read fasta file
    """
    with open(fasta_file, 'r') as f:
        description = f.readline().strip()
        sequence = ''
        for line in f:
            if line.startswith('>'):
                break
            sequence += line.strip()
    return sequence, description

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

    with open(join(output_dir, batch['name']+'.pdb'), 'w') as f:
        f.write(pdb_str)


def main(args):
    """main function"""
    ### create model
    af2_model_name="seq21_pair48_l8_512_vio2"
    af2_model_config = config.model_config(af2_model_name)


    with open('./data/processed/batchs_nature_unbounded.pkl', 'rb') as file:
        unbounded=pickle.load(file)
    with open('./data/processed/batchs_nature_bounded.pkl', 'rb') as file:
        bounded=pickle.load(file)
    with open('./data/bounded_test.pkl', 'rb') as file:
        bounded_test=pickle.load(file)
    with open('./data/unbounded_test.pkl', 'rb') as file:
        unbounded_test=pickle.load(file)

    model = Run_paddleModel(af2_model_config)

    if args.bounded:
        model.load_params("./model/Bounded_model.pdparams")
    else:
        model.load_params("./model/Unbounded_model.pdparams")

    print("bert_model size:", get_model_parameter_size(model.model_bert))
    print("fold size:", get_model_parameter_size(model)-get_model_parameter_size(model.model_bert))
    print("model size:", get_model_parameter_size(model))

    ### make predictions
    af2_model_config.data.eval.delete_msa_block = False

    if args.bounded:
        batchs = copy.deepcopy(bounded)
        testset=copy.deepcopy(bounded_test)
    else:
        batchs = copy.deepcopy(unbounded)
        testset=copy.deepcopy(unbounded_test)


    model.eval()

    l=len(batchs)

    output_dir = "./output_stru/testset"
    os.makedirs(output_dir, exist_ok=True)

    while 1:
        for i in range(l):
            batch=copy.deepcopy(batchs[i])
            if batch['name'] not in testset:
                continue
            batch['feat'] = tree_map(lambda v: paddle.to_tensor(v), batch['feat'])
            batch['label'] = tree_map(lambda v: paddle.to_tensor(v), batch['label'])
            with paddle.no_grad():
                results= model(batch, bounded=batch['bounded'])
            postprocess(batch, results, output_dir)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--bounded", type=bool)
    args = parser.parse_args()

    args.output_dir="./output_stru"
    args.bounded=True

    main(args)
