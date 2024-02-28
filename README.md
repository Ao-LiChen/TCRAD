# TCRAD
## Introduction 
TCRAD: A Deep Learning Based Pipeline for TCR Auto Design and Optimization. 

Publication: 

The model consists primarily of three modules:
* Sequence Generation Module
* Sequence Filtration Module
* Structure Prediction Module
 
![Figure 1](Model.png)

## Dependencies
1. Install Anaconda or Miniconda.
2. Create a new conda environment:
```bash
conda create -n TCRAD python=3.8

conda activate TCRAD
```
3. Install PyTorch and PaddlePaddle:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# For sequence generation only, you can skip the installation of the PaddlePaddle framework.
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.6 -c
```
4. Install the following packages::
```bash
conda install scipy==1.8.0 scikit_learn==1.1.3 pandas==1.3.4 absl-py==0.13.0 biopython==1.79 dm-haiku==0.0.4 dm-tree==0.1.6 chex==0.0.7 docker==5.0.0 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 pandas==1.3.4 scipy==1.7.0 tensorflow-cpu==2.6.0
```
**Note**:Please ensure that the PyTorch version and Paddle version are compatible with your CUDA version. You can choose the appropriate versions to install to match your environment configuration.

## Data and Model Weights
The datasets and model weights mentioned in the paper: [Data and Model](https://zenodo.org/records/10715856)


## Designing
Input Peptide sequence and output the generated and filtered CDR3B sequences (.csv) to the output_seq folder.

    python peptide_to_cdr3B.py --Peptide SLLMWITQC --minibatchsize 1000

    Required:
        --Peptide: the peptide sequence you want to design the TCR for.
        --minibatchsize: configure the size of the cache space. It has been tested that 16GB of memory corresponds to 2500.

#### * Note : Generating all 20 CDR3B positions can be time-consuming. Therefore, it is recommended to use mutation by inputting "--------" to fix the CDR3B length to 8 positions, which can be more efficient.

    
## Mutation 
Input the Peptide sequence and the CDR3B positions for mutation optimization, and output the mutated and filtered CDR3B sequences (.csv) to the output_seq folder.

Command 1:  

    python peptide_to_cdr3B_mutation.py --Peptide SLLMWITQC --minibatchsize 1000 --CDR3B CASSYV--TGELFF   
  
    Required:
        --Peptide: the peptide sequence you want to design the TCR for.
        --minibatchsize: configure the size of the cache space. It has been tested that 16GB of memory corresponds to 2500.
        --CDR3B: the CDR3B sequence you want to mutate,the "-" represents the position to be mutated.

Command 2:  

    python peptide_to_cdr3B_mutation.py --Peptide SLLMWITQC --minibatchsize 1000 --CDR3B CASSYVGNTGELFF --POSITION 7 8  
  
    Required:
        --Peptide: the peptide sequence you want to design the TCR for.
        --minibatchsize: configure the size of the cache space. It has been tested that 16GB of memory corresponds to 2500.
        --CDR3B: the CDR3B sequence you want to mutate.
        --POSITION: the position of the CDR3B you want to mutate.

#### * Note :The model will finally output three CSV files: .csv, filtered.csv, soft_filtered.csv. They respectively represent all generated sequences, sequences after high-threshold filtering, and sequences after medium-threshold filtering by the sequence filtering module.

## Structure Prediction
Input the CDR3B sequence and output the predicted Unbinding CDR3 structure to the output_stru folder.

    python CDR3_stru_generate.py --CDR3 CATSALGDTQYF --Peptide SLLMWITQC
    Required:
        --CDR3: the CDR3 sequence you want to predict the structure for.
        --Peptide: the peptide sequence you want to design the TCR for.

Used for structural prediction of CDR3 in the test set:

    python CDR3_stru_generate_TESTSET.py --bounded True
    Required:
        --bounded: Indicates whether it is on the bounded test set or the unbounded test set.


## Citation
## Contacts
