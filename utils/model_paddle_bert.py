from alphafold_paddle.model import modules

import math
import paddle
import numpy as np
import paddle.nn as nn
from utils import *
from random import *
import paddle.nn.functional as F



class Run_paddleModel(nn.Layer):
    """
    RunModel
    """
    def __init__(self, af2_model_config):
        super(Run_paddleModel, self).__init__()
        self.af2_model_config = af2_model_config
        self.d_model=512
        self.n_layers=8
        self.n_heads=6
        self.model_bert = BERT(n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, maxlen=21, vocab_size=24)
        self.model_bert_binding = BERT(n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, maxlen=37, vocab_size=24,n_segments=2)

        self.freeze_tape = False
        self._init_encoder()

        # channel_num = {k: v.shape[-1] for k, v in self.batch.items()}
        # pylint: disable=
        channel_num = {'aatype': 106, 'residue_index': 106, 'seq_length': 1, 
            'is_distillation': 1, 'seq_mask': 106, 'msa_mask': 106, 
            'msa_row_mask': 512, 'random_crop_to_size_seed': 2, 
            'atom14_atom_exists': 14, 'residx_atom14_to_atom37': 14, 
            'residx_atom37_to_atom14': 37, 'atom37_atom_exists': 37, 
            'extra_msa': 106, 'extra_msa_mask': 106, 'extra_msa_row_mask': 1024, 
            'bert_mask': 106, 'true_msa': 106, 'extra_has_deletion': 106, 
            'extra_deletion_value': 106, 'msa_feat': 49, 'target_feat': 22}
        self.alphafold = modules.AlphaFold(channel_num, af2_model_config.model)

    def _init_encoder(self):
        self.tape_single_linear = nn.Linear(
                self.d_model,
                self.af2_model_config.model.embeddings_and_evoformer.msa_channel)

        self.tape_peptide_single_linear = nn.Linear(
                self.d_model * 2,
                self.af2_model_config.model.embeddings_and_evoformer.msa_channel)

        weight_out_dim = self.n_layers * self.n_heads

        self.tape_pair_linear = nn.Linear(
                weight_out_dim,
                self.af2_model_config.model.embeddings_and_evoformer.pair_channel)



    def _forward_torch(self, batch,bounded):

        #tape_results= paddle.to_tensor(batch["bert_results"],dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
        #attn_weight= paddle.to_tensor(batch["bert_attn_weight"],dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)


        if bounded:
            #peptide_result = paddle.to_tensor(batch["bert_result_peptide"], dtype='float32', place=paddle.CUDAPlace(0),stop_gradient=False)
            peptide_result = paddle.mean(batch["bert_result_peptide"], axis=2, keepdim=True)
            peptide_result = paddle.expand_as(peptide_result, batch["bert_results"])
            tape_results = paddle.concat([batch["bert_results"], peptide_result], axis=-1)

            tape_single = self.tape_peptide_single_linear(tape_results)   # (b, num_recycle, num_res, msa_channel)
        else:
            tape_single = self.tape_single_linear(batch["bert_results"])

        tape_pair = self.tape_pair_linear(batch["bert_attn_weight"])  # (b, num_recycle, num_res, num_res, pair_channel)
        batch['feat'].update(tape_single=tape_single, tape_pair=tape_pair)

        return batch
    
    def forward(self, batch, compute_loss=False,bounded=False):
        """
        all_atom_mask: (b, N_res, 37)
        """
        if not bounded:
            seq=batch['seq']
            seg=batch['seg']
            flagseq=seq[0].reshape(1,-1)
            left = np.count_nonzero(flagseq[:, :10] == 22)
            right = np.count_nonzero(flagseq[:, 10:] == 22)

            seq = paddle.to_tensor(seq, dtype='int64', stop_gradient=False)
            seg = paddle.to_tensor(seg, dtype='int64', stop_gradient=False)

            tape_results, attn_weight = self.model_bert(seq, seg)
            num_recycle = batch['feat']['aatype'].shape[1]

            tape_results=tape_results.unsqueeze(1)
            attn_weight=attn_weight.unsqueeze(1)

            tape_results = paddle.tile(tape_results, repeat_times=(1, num_recycle, 1, 1))
            attn_weight = paddle.tile(attn_weight, repeat_times=(1, num_recycle, 1, 1, 1))


            # tape_results = tape_results[:, :, left:-(right + 1)]  # (b, num_recycle, num_res, d1)
            # attn_weight = attn_weight[:, :, :, left:-(right + 1), left:-(right + 1)].transpose(
            #     [0, 1, 3, 4, 2])  # (b, num_recycle, num_re

            # 对 tape_results 进行切片
            tape_results = paddle.slice(tape_results, axes=[2], starts=[left], ends=[tape_results.shape[2] - right - 1])
            # 对 attn_weight 进行切片和转置
            attn_weight = paddle.slice(attn_weight, axes=[3, 4], starts=[left, left],
                                       ends=[attn_weight.shape[3] - right - 1, attn_weight.shape[4] - right - 1])

            attn_weight = paddle.transpose(attn_weight, perm=[0, 1, 3, 4, 2])

            batch['bert_results'] = tape_results
            batch['bert_attn_weight'] = attn_weight
        else:
            seq=np.array(batch['seq'])
            seg=np.array(batch['seg'])
            flagseq=seq.reshape(1,-1)
            flagseg=seg.reshape(1,-1)
            left = np.count_nonzero(flagseq[:, 16:26] == 22)
            right = np.count_nonzero(flagseq[:, 26:] == 22)

            flagseq = paddle.to_tensor(flagseq, dtype='int64', stop_gradient=False)
            flagseg = paddle.to_tensor(flagseg, dtype='int64', stop_gradient=False)

            tape_results, attn_weight = self.model_bert_binding(flagseq, flagseg)
            num_recycle = batch['feat']['aatype'].shape[1]

            tape_results=tape_results.unsqueeze(1)
            attn_weight=attn_weight.unsqueeze(1)

            tape_results = paddle.tile(tape_results, repeat_times=(1, num_recycle, 1, 1))
            attn_weight = paddle.tile(attn_weight, repeat_times=(1, num_recycle, 1, 1, 1))
            peplen=len(batch['peptide'])
            tape_results_cdr3 = tape_results[:, :, 16 + left:-(right + 1)]  # (b, num_recycle, num_res, d1)
            #tape_results_peptide = tape_results[:, :, :peplen]  # (b, num_recycle, num_res, d1)

            attn_weight = attn_weight[:, :, :, 16 + left:-(right + 1), 16 + left:-(right + 1)].transpose([0, 1, 3, 4, 2])  # (b, num_recycle, num_re
            batch['bert_results'] = tape_results_cdr3
            batch['bert_attn_weight'] = attn_weight
            batch['bert_result_peptide'] = paddle.concat([paddle.unsqueeze(tape_results[:, :, -1, :],axis=2),paddle.unsqueeze(tape_results[:, :, 15, :],axis=2)],axis=2)


        #调用tape模型(Bert模型魔改)
        batch = self._forward_torch(batch,bounded)

        # 基于提取出来的特征给Evoformer和Structure_Model做结构预测
        res = self.alphafold(
            batch['feat'],
            batch['label'],
            ensemble_representations=True,
            return_representations=True,
            compute_loss=compute_loss)

        if compute_loss:
            results, loss = res
            # if self.loss_rescale_with_n_res:
            #     N_res = paddle.sum(batch['label']['all_atom_mask'][:, :, 0], 1)
            #     loss = loss * paddle.sqrt(paddle.cast(N_res, 'float32'))
            return results, loss.mean()
        else:
            return res
    
    def load_bert_params(self, bert_init_model):
        """tbd"""
        if not bert_init_model is None and bert_init_model != "":
            print(f"Load model from {bert_init_model}")
            flag=paddle.load(bert_init_model)
            self.model_bert.set_state_dict(flag)

    def load_binding_bert_params(self, bert_init_model):
        """tbd"""
        if not bert_init_model is None and bert_init_model != "":
            print(f"Load model from {bert_init_model}")
            #flag=paddle.load(bert_init_model)
            self.model_bert_binding.set_state_dict(bert_init_model)

    
    def load_params(self, init_model):
        """tbd"""
        if not init_model is None and init_model != "":
            print(f"Load model from {init_model}")
            flag=paddle.load(init_model)
            self.set_state_dict(flag)
    
    def save_params(self, param_path):
        paddle.save(self.state_dict(), param_path)








def get_attn_pad_mask(seq_q):

    batch_size, seq_len = seq_q.shape
    # eq(22) is PAD token
    pad_attn_mask = paddle.equal(seq_q, 22).unsqueeze(1)

    pad_attn_mask=paddle.tile(pad_attn_mask,[1, seq_len, 1])

    return pad_attn_mask  # [batch_size, seq_len, seq_len]



def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    #return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


class Embedding(nn.Layer):
    def __init__(self, d_model, maxlen, n_segments, vocab_size, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.shape[1]
        if self.device == "cuda":
            pos = paddle.arange(seq_len).cuda()
        else:
            pos = paddle.arange(seq_len).to(self.device)

        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] ->

        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Layer):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
    def forward(self, Q, K, V, attn_mask):
        scores = paddle.matmul(Q, K.transpose([0,1, 3, 2])) / np.sqrt(self.d_model * 2) # scores : [batch_size, n_heads, seq_len, seq_len]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.

        out = paddle.full(scores.shape, -1e9, scores.dtype)
        scores = paddle.where(attn_mask, out, scores)

        attn = nn.functional.softmax(scores, axis=-1)

        context = paddle.matmul(attn, V)
        return context,attn


class MultiHeadAttention(nn.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model * 2 * n_heads) #dk
        self.W_K = nn.Linear(d_model, d_model * 2 * n_heads) #dk
        self.W_V = nn.Linear(d_model, d_model * 2 * n_heads) #dv
        self.d_model = d_model
        self.n_heads = n_heads
        self.liner = nn.Linear(n_heads * self.d_model * 2, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.shape[0]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).reshape([batch_size, -1, self.n_heads, self.d_model * 2]).transpose([0, 2, 1, 3])  # 修改这里
        k_s = self.W_K(K).reshape([batch_size, -1, self.n_heads, self.d_model * 2]).transpose([0, 2, 1, 3])
        v_s = self.W_V(V).reshape([batch_size, -1, self.n_heads, self.d_model * 2]).transpose([0, 2, 1, 3])

        attn_mask = attn_mask.unsqueeze(1).tile([1, self.n_heads, 1, 1]) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context,atten = ScaledDotProductAttention(self.d_model)(q_s, k_s, v_s, attn_mask)
        context = context.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.n_heads * self.d_model * 2]) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.liner(context)
        return self.layernorm(output + residual),atten # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Layer):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))+x


class EncoderLayer(nn.Layer):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs,attn_mask = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs,attn_mask


class BERT(nn.Layer):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, \
                 maxlen=72, n_segments=1, vocab_size=24, device="cuda"):
        super(BERT, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(d_model=d_model, maxlen=maxlen, \
                                   n_segments=n_segments, vocab_size=vocab_size ,device=device)
        #self.layers = [EncoderLayer(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        self.layers=paddle.nn.LayerList([EncoderLayer(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p=0.6)
        self.activ2 = gelu

        # fc2 is shared with embedding layer
        #embed_weight = self.embedding.tok_embed.weight
        #self.fc2 = nn.Linear(d_model, vocab_size, bias_attr=False)
        #flag = paddle.transpose(embed_weight, perm=[1, 0])
        #self.fc2.weight=flag

        self.fc2 = paddle.nn.Linear(d_model, vocab_size, bias_attr=False)
        #self.fc2._weight_attr = paddle.ParamAttr(name='shared_weight')




        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(128,32),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def paddle_gather(self,x, dim, index):
        index_shape = index.shape
        index_flatten = index.flatten()
        if dim < 0:
            dim = len(x.shape) + dim
        nd_index = []
        for k in range(len(x.shape)):
            if k == dim:
                nd_index.append(index_flatten)
            else:
                reshape_shape = [1] * len(x.shape)
                reshape_shape[k] = x.shape[k]
                x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
                x_arange = x_arange.reshape(reshape_shape)
                dim_index = paddle.expand(x_arange, index_shape).flatten()
                nd_index.append(dim_index)
        ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
        paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
        return paddle_out

    def forward(self, input_ids, segment_ids, masked_pos='',mask="False"):
        attention_matrices=[]

        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output,attn_mask = layer(output, enc_self_attn_mask)
            attention_matrices.append(attn_mask)
        attention = paddle.concat(attention_matrices, axis=1)

        # prediction mode
        if mask == "False":
            return output,attention

        # training model
        else:
            masked_pos_expanded = paddle.unsqueeze(masked_pos, axis=2)
            masked_pos_tiled = paddle.tile(masked_pos_expanded, [1, 1, self.d_model])
            masked_pos_1d = paddle.reshape(masked_pos, shape=[-1])

            #masked_pos = masked_pos[:, :, None].expand(len(input_ids), -1, self.d_model) # [batch_size, max_pred, d_model]
            h_masked = self.paddle_gather(output, 1,masked_pos_tiled) # masking position [batch_size, max_pred, d_model]#根据masked_pos的坐标，提取出output对应的256维向量
            h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
            #logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
            logits_lm = paddle.matmul(h_masked, self.embedding.tok_embed.weight, transpose_y=True)
            return logits_lm, output

