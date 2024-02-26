import argparse
import torch.nn as nn
from embeding import *
from utils.utils_seq import *
from bert_Unilm_mask import BERT
from bert_CDR3_P import BERT as BERT_Bind
from bert_discriminator import BERT as BERT_Discriminator
import torch.nn.functional as F
import pandas as pd

parser = argparse.ArgumentParser()

# File dir
parser.add_argument('--model_dir', type=str,default="./model/Unilm_6MASK.pt")

parser.add_argument('--bindding_model_dir', type=str, default="./model/CDR3_P_Bindding.pt")

parser.add_argument('--discriminator_model_load_dir', type=str, default="./model/discriminator_model.pt")

# Hyperparameters
parser.add_argument('--n_layers', type=int, default=8, help='number of encoder layers')
parser.add_argument('--d_model', type=int, default=512, help='number of embedding dimention')

#Input Sequence
parser.add_argument('--Peptide', type=str, default="SLLMWITQC")
parser.add_argument('--minibatchsize', type=int, default=1000, help='2500 corresponds to 16GB of GPU memory')

# GPUs
parser.add_argument('--GPUs', type=int, default=1, help='num of GPUs used in this task')

args = parser.parse_args()

# BERT Parameters
maxlen = 37  # max tokens of pmhc sequence
n_layers = args.n_layers  # num of transformer encoder layers
n_heads = 6  # num of muti self-attention heads
d_model = args.d_model  # embedding dimention
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = d_model * 2  # dimension of K(=Q), V
n_segments = 2  # num of types of segment tokens
vocab_size = 24  # type of tokens

# Determine if GPU is available
if not torch.cuda.is_available():
    raise ValueError("No available GPU")

# Determine the number of available GPUs
GPUs_used = args.GPUs
GPUs_avail = torch.cuda.device_count()
if GPUs_used > GPUs_avail:
    raise ValueError("Available GPU({}) is less than the input value".format(GPUs_avail))

p2n = AA
n2p = {}
for key, value in p2n.items():
    n2p[value] = key

p2n['<MASK>'] = 21
n2p[21] = '<MASK>'
p2n['-'] = 22
n2p[22] = '-'
p2n['<EOS>'] = 23
n2p[23] = '<EOS>'


# Map amino acids to serial numbers
def aa_to_index(cdr3):
    new_cdr3 = []
    for aa in cdr3:
        new_cdr3.append(AA[aa])
    return new_cdr3


# Initialization Model
model_G = BERT(n_layers=n_layers, d_model=d_model, n_heads=n_heads, \
               maxlen=maxlen, n_segments=n_segments, vocab_size=vocab_size)

state = torch.load(args.model_dir)
new_state = {}
for key, value in state.items():
    new_key = key[7:]
    new_state[new_key] = value
model_G.load_state_dict(new_state)

del state, new_state

# Run the main program
if __name__ == "__main__":

    # Data Parallelism
    model_G = nn.DataParallel(model_G, list(range(GPUs_used)), output_device=0)

    model_G.cuda()
    # Initializing hyperparameters

    torch.cuda.empty_cache()
    model_G.eval()

    peptide = args.Peptide
    minibatchsize = args.minibatchsize

    input_ids = aa_to_index(peptide) + [22] * (15 - len(peptide)) + [23] + [21] * 20 + [23]
    cand_maked_pos = [25, 23, 24, 29, 30, 22, 27, 31, 26, 28, 21, 20, 32, 19, 18, 33, 17, 34, 16,
                      35]
    input_ids = torch.LongTensor(input_ids)
    input_ids = input_ids.view(-1, 37)
    all = []

    segment_ids = [0] * 16 + [1] * 21
    segment_ids = torch.LongTensor(segment_ids)
    segment_ids = segment_ids.view(-1, 37).cuda()
    while 1:
        masked_pos = []
        pos = cand_maked_pos[0]
        masked_pos.append(pos)
        masked_pos = torch.LongTensor(masked_pos)
        masked_pos = masked_pos.view(1, -1).cuda()
        input_ids[:, pos] = 21  # make mask

        if len(input_ids) < minibatchsize:
            with torch.no_grad():
                # print(len(input_ids))
                # print(input_ids.view(-1, 37).shape)
                # print(segment_ids.expand_as(input_ids).shape)
                # print(masked_pos.expand(len(input_ids), 1).shape)
                logits_lm, _ = model_G(input_ids.cuda(), segment_ids.expand_as(input_ids).cuda(), "attention",
                                       masked_pos.expand(len(input_ids), 1).cuda())
                # print(len(logits_lm))
        else:
            input_ids_chunks = [input_ids[i:i + minibatchsize] for i in range(0, len(input_ids), minibatchsize)]
            logits_lm_list = []
            for input_ids_chunk in input_ids_chunks:
                with torch.no_grad():
                    # print(input_ids_chunk.view(-1, 37).shape)
                    # print(segment_ids.shape)
                    # print(masked_pos.shape)

                    logits_lm, _ = model_G(input_ids_chunk.view(-1, 37).cuda(),
                                           segment_ids.expand_as(input_ids_chunk).cuda(), "attention",
                                           masked_pos.expand(len(input_ids_chunk), 1).cuda())
                    # print(len(logits_lm))
                logits_lm_list.append(logits_lm.cpu())

            logits_lm = torch.cat(logits_lm_list, dim=0)
            del logits_lm_list

        logits_lm = logits_lm.cpu()

        # 分析处理mask结果
        # for masked token prediction
        temperature = 1
        kk = F.softmax(logits_lm.view(-1, 24) / temperature, dim=1)
        l = len(kk)
        right = 0
        flag = 0
        print("MASK position:" + str(pos))

        input_ids = input_ids.view(-1, 37)
        for i in range(l):

            f1 = kk[i].tolist()
            f3 = sorted(f1, reverse=True)

            # 采用TOPP策略，累加到0.8
            TOPP = 0
            for j in range(24):
                TOPP += f3[j]
                if TOPP >= 0.8:
                    break
            candidate = f3[:j + 1]
            maxpre = max(candidate)

            ori_candidate = candidate
            # 采用TOPK策略,分段
            if pos in [25, 23, 24]:
                candidate = [_ for _ in candidate if (_ > 0.05)][:6]
            elif pos in [29, 30, 22, 27, 31, 26, 28, 21]:
                candidate = [_ for _ in candidate if (_ > 0.05)][:3]
            else:
                candidate = [_ for _ in candidate if (_ > maxpre * 0.5) and (_ > 0.05)][:2]
            if candidate == []:
                candidate = ori_candidate[:2]
            # 采用topK策略
            ans = []
            for each in candidate:
                ans.append([n2p[f1.index(each)], each])

            current_device = torch.cuda.current_device()
            indices = [0]
            for _ in range(args.GPUs):
                indices.append(indices[-1] + input_ids.size(0))

            offset = i + flag - indices[current_device]

            repeated_rows = [input_ids[offset, :]] * len(ans)
            FLAG = torch.cat(repeated_rows, dim=0).view(-1, 37)
            for i in range(len(ans)):
                FLAG[i][pos] = p2n[ans[i][0]]

            input_ids = torch.cat((input_ids[:offset, :], FLAG, input_ids[offset + 1:, :]), dim=0)
            flag += len(ans) - 1

        # input_ids = input_ids.view(-1, 37)
        print(input_ids.size(0))

        l = len(input_ids)
        # 处理填入MASK位置的值怎么选

        # 处理下一个mask位置
        cand_maked_pos = cand_maked_pos[1:]

        if not len(cand_maked_pos):
            break
    # print(all)
    C = input_ids.tolist()
    l = len(input_ids)
    Generation = []

    for i in range(l):
        C[i][16:37] = [n2p[x] for x in C[i][16:37]]
        k = C[i][16:36]

        # 使用join方法连接
        seq = ''.join(k)
        Generation.append(seq)

    del model_G

    model_Bind = BERT_Bind(n_layers=n_layers, d_model=d_model, n_heads=n_heads, \
                           maxlen=maxlen, n_segments=n_segments, vocab_size=vocab_size)

    state = torch.load(args.bindding_model_dir)
    new_state = {}
    for key, value in state.items():
        new_key = key[7:]
        new_state[new_key] = value
    model_Bind.load_state_dict(new_state)
    del state, new_state

    # Data Parallelism
    model_Bind = nn.DataParallel(model_Bind, list(range(GPUs_used)))

    model_Bind.cuda()

    torch.cuda.empty_cache()
    model_Bind.eval()


    input_ids_chunks = [input_ids[i:i + minibatchsize] for i in range(0, len(input_ids), minibatchsize)]
    logits_clsf_list = []
    for input_ids_chunk in input_ids_chunks:
        with torch.no_grad():
            logits_lm, logits_clsf, _ = model_Bind(input_ids_chunk.cuda(),
                                                   segment_ids.expand_as(input_ids_chunk).cuda(),
                                                   masked_pos.expand(len(input_ids_chunk), 1).cuda())
        logits_clsf_list.append(logits_clsf.cpu())

        logits_clsf = torch.cat(logits_clsf_list, dim=0)

    del logits_lm

    del model_Bind

    model_Discriminator = BERT_Discriminator(n_layers=8, d_model=512, n_heads=6, maxlen=21, n_segments=1, vocab_size=24)

    state = torch.load(args.discriminator_model_load_dir)
    new_state = {}
    for key, value in state.items():
        new_key = key[7:]
        new_state[new_key] = value
    model_Discriminator.load_state_dict(new_state)
    del state, new_state

    # Data Parallelism
    model_Discriminator = nn.DataParallel(model_Discriminator, list(range(GPUs_used)))
    model_Discriminator.cuda()
    torch.cuda.empty_cache()
    model_Discriminator.eval()

    logits_dis_list = []
    segment_ids = torch.LongTensor([0] * 21)
    for input_ids_chunk in input_ids_chunks:
        with torch.no_grad():
            input_ids_chunk = input_ids_chunk[:, 16:37]
            logits_dis, _ = model_Discriminator(input_ids_chunk.cuda(), segment_ids.expand_as(input_ids_chunk).cuda())
        logits_dis_list.append(logits_dis.cpu())

    logits_dis = torch.cat(logits_dis_list, dim=0)

    logits = logits_clsf + (1 - logits_dis) * 2

    # Extract the sequences with the highest binding prediction values
    logits_clsf = logits_clsf.view(-1).tolist()
    logits_dis = logits_dis.view(-1).tolist()

    data = {
        'CDR3': Generation,
        'clsf': logits_clsf,
        'dis': logits_dis
    }
    df = pd.DataFrame(data)

    filtered_df = df[(df['dis'] < 0.26) & (df['clsf'] > 0.68)]

    soft_filtered_df = df[(df['dis'] < 0.56) & (df['clsf'] > 0.28)]

    df.to_csv('output_seq/' + peptide + '_CDR3B_Generated.csv', index=False)
    if filtered_df.shape[0] > 0:
        filtered_df.to_csv('output_seq/' + peptide + '_CDR3B_Generated_filtered.csv', index=False)
    if soft_filtered_df.shape[0] > 0:
        soft_filtered_df.to_csv('output_seq/' + peptide + '_CDR3B_Generated_soft_filtered.csv', index=False)

