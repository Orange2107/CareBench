
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
import math
import torch
from torch.nn.modules import TransformerEncoderLayer

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
IGNORE_ID = -1

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(
            hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            print('groundtruth: %s' % out_dic['text'])
            print('prediction : %s' % out_dic['rec_text'])

    return new_js

# -- Transformer Related --
# import torch

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    # print("padded_input: ", padded_input.shape)
    # print("input_lengths: ", input_lengths)
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

def get_multi_attn_pad_mask(padded_input, input_lengths, expand_length, additional_cls_mask=None):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_masks = [get_non_pad_mask(padded_input[i], input_lengths=input_lengths[i]) for i in range(len(padded_input))] 
    non_pad_masks = torch.cat(non_pad_masks, axis = 1)
    if additional_cls_mask is not None:
        non_pad_masks = torch.cat((additional_cls_mask, non_pad_masks), axis = 1)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_masks.squeeze(-1).lt(1)
    # attn_mask = pad_mask.unsqueeze(1).expand(-1, sum(expand_length)+1, -1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)

    return attn_mask


def get_cross_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


def get_decoder_self_attn_mask(seq_k, seq_q, pad_id):
    """ For masking the decoder self attention """
    def _get_attn_key_pad_mask(seq_k, seq_q, pad_id):
        """ For masking out the padding part of key sequence. """
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(pad_id)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    def _get_subsequent_mask(inputs):
        """ Makes subsequent masking """
        batch_size, seq_length = inputs.size()
        subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=inputs.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # BxTxT

        return subsequent_mask.bool()

    return _get_attn_key_pad_mask(seq_k, seq_q, pad_id) | _get_subsequent_mask(seq_k)



class BimodalTransformerEncoder_MBT(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            batch_size: int,
            n_modality: int,
            bottlenecks_n: int,
            
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 10000,
            txt_idx: int = 2,
            mbt_bottlenecks_type: str = 'skip',
            use_pe: list = [True, True],
            mask: list = [True, True]):
        super(BimodalTransformerEncoder_MBT, self).__init__()

        self.mbt_bottlenecks_type = mbt_bottlenecks_type
        self.use_pe = use_pe
        self.n_modality = 2
        self.num_heads=n_head
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
                
        # CLASSIFICATION TOKENS
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                nhead = n_head,
                dim_feedforward = d_ff,
                dropout = dropout,
                batch_first=True
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        batch_size=enc_outputs[0].size(0)
        cls_token_per_modality = [cls_token.repeat(batch_size, 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(batch_size, 1, 1)
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_input], axis=1) for idx, enc_input in enumerate(enc_outputs)]
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        for n_modal in range(self.n_modality):
            varying_lengths[n_modal] += 1
            fixed_lengths[n_modal] += 1
            if n_modal == self.txt_idx:
                    varying_lengths[n_modal][varying_lengths[n_modal] == 3] = 0
            if self.mask[n_modal]:
                self_attn_masks.append(get_attn_pad_mask(enc_inputs[n_modal], varying_lengths[n_modal], fixed_lengths[n_modal]))
            else:
                self_attn_masks.append(None)
                
        # if fusion_idx is not None:
        #     self.fusion_idx = fusion_idx
        
        idx_order = torch.arange(batch_size, dtype=torch.long, device=enc_outputs[0].device)
            
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) +
                    self.positional_encoding(enc_inputs[idx])
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            
            bottleneck_outputs = []
            for modal_idx, enc_layer in enumerate(enc_layers):
                b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], axis=1) #bottleneck, cls, input
                if len(bottleneck_self_attn_masks) < self.n_modality:
                    if self.mask[modal_idx]:
                        b_mask = get_attn_pad_mask(b_enc_output, varying_lengths[modal_idx]+self.bottlenecks_n, b_enc_output.size(1))
                        # (B, L, L)->(B*num_heads, L, L)
                        b_mask = b_mask.repeat_interleave(self.num_heads, dim=0) 
                        bottleneck_self_attn_masks.append(b_mask)
                    else:
                        bottleneck_self_attn_masks.append(None)
                # enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                enc_output = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                        
                bottleneck_outputs.append(enc_output[:, :self.bottlenecks_n, :])
                enc_output = enc_output[:, self.bottlenecks_n:, :]
                enc_outputs.append(enc_output)
                
            bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
            bottlenecks_bi_mean = torch.mean(bottleneck_outputs_stack, dim=0)
            all_bottleneck_stack = torch.stack([bottlenecks_bi_mean, bottleneck_outputs_stack[0,:,:,:]])
            # missing: 0-> ehr+cxr, 1->ehr
            bottlenecks = all_bottleneck_stack[missing, idx_order, :, :]
            
        return enc_outputs, 0

class PositionalEncoding(nn.Module):
    """Positional encoding with mask support"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        # x = x + self.pe[:, :x.size(1)]
        # return self.dropout(x)
class TrimodalTransformerEncoder_MBT(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
                 num_final_classes: int,
            batch_size: int,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 10000,
            resbottle: bool = False,
            txt_idx: int = 2,
            vsltonly: int = 0,
            mbt_bottlenecks_type: str = 'skip',
            use_pe: list = [True, True, True],
            mask: list = [True, False, True]):
        super(TrimodalTransformerEncoder_MBT, self).__init__()
        self.num_final_classes=num_final_classes
        self.vsltonly = vsltonly
        self.mbt_bottlenecks_type = mbt_bottlenecks_type
        self.use_pe = use_pe
        self.n_modality = n_modality
        self.fusion_idx = fusion_startidx
        self.txt_idx = txt_idx
        self.n_layers = n_layers
        self.d_model = d_model
        self.bottlenecks_n = bottlenecks_n
        self.mask = mask
        self.resbottle = resbottle
        self.num_heads=8
        
        self.idx_order = torch.arange(0, batch_size).type(torch.LongTensor)
        
        self.layer_norms_after_concat = nn.LayerNorm(self.d_model)
                
        # CLASSIFICATION TOKENS
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, self.num_final_classes, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, enc_outputs, fixed_lengths = None, varying_lengths = None, return_attns = False, fusion_idx = None, missing = None):
        cls_token_per_modality = [cls_token.repeat(enc_outputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(enc_outputs[0].size(0), 1, 1)
        enc_inputs = [torch.cat([cls_token_per_modality[idx], enc_input], axis=1) for idx, enc_input in enumerate(enc_outputs)]
        
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        for n_modal in range(self.n_modality):
            varying_lengths[n_modal] += self.num_final_classes
            fixed_lengths[n_modal] += self.num_final_classes
            if self.mask[n_modal]:
                # print("1 ", enc_inputs[n_modal].shape)
                # print("2 ", varying_lengths[n_modal])
                # print("3 ", fixed_lengths[n_modal])
                self_attn_masks.append(get_attn_pad_mask(enc_inputs[n_modal], varying_lengths[n_modal], fixed_lengths[n_modal]))
            else:
                self_attn_masks.append(None)
                
        if fusion_idx is not None:
            self.fusion_idx = fusion_idx
            
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx]) +
                    self.positional_encoding(enc_inputs[idx])
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](enc_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            if idx < self.fusion_idx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx].repeat_interleave(self.num_heads, dim=0))
                    enc_outputs.append(enc_output)      
                    
            else:
                bottleneck_outputs = []
                if self.resbottle:
                    res_bottles = bottlenecks

                for modal_idx, enc_layer in enumerate(enc_layers):
                    b_enc_output = torch.cat([bottlenecks, enc_inputs[modal_idx]], axis=1) #bottleneck, cls, input
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[modal_idx]:
                            b_mask = get_attn_pad_mask(b_enc_output, varying_lengths[modal_idx]+self.bottlenecks_n, b_enc_output.size(1))
                            bottleneck_self_attn_masks.append(b_mask)
                        else:
                            bottleneck_self_attn_masks.append(None)
                            
                    enc_output = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx].repeat_interleave(self.num_heads, dim=0))
                    bottleneck_outputs.append(enc_output[:, :self.bottlenecks_n, :])
                    enc_output = enc_output[:, self.bottlenecks_n:, :]
                    enc_outputs.append(enc_output)
                    
                # bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
                # bottlenecks_tri_mean = torch.mean(bottleneck_outputs_stack, dim=0)
                # bottlenecks_vslttxt_mean = torch.mean(torch.stack([bottleneck_outputs_stack[0,:,:,:], bottleneck_outputs_stack[2,:,:,:]]), dim=0)
                # bottlenecks_vsltimg_mean = torch.mean(bottleneck_outputs_stack[:2,:,:,:], dim=0)
                # all_bottleneck_stack = torch.stack([bottlenecks_tri_mean, bottlenecks_vsltimg_mean, bottlenecks_vslttxt_mean, bottleneck_outputs_stack[0,:,:,:]])
                
                # print("missing: ", missing)
                # print("missing: ", missing.shape)
                # print("all_bottleneck_stack: ", all_bottleneck_stack.shape)
                # print("self.idx_order: ", self.idx_order)
                # print("self.idx_order: ", self.idx_order.shape)
                
                # bottlenecks = all_bottleneck_stack[missing, self.idx_order, :, :]
                # # print("bottlenecks: ", bottlenecks.shape) torch.Size([32, 4, 256])
                # if self.resbottle:
                #     bottlenecks = torch.mean(torch.stack([bottlenecks,res_bottles]), dim=0)
                
                # bottlenecks = torch.where(missing[1].unsqueeze(1).unsqueeze(1) == 0, bottleneck_outputs[0], bottlenecks_mean)
                # bottlenecks = torch.where(varying_lengths[1].unsqueeze(1).unsqueeze(1) == 0, bottleneck_outputs[0], bottlenecks_mean)
                
        return enc_outputs, 0
    
class MBTEncoder(nn.Module):
    """
    Based on "Attention Bottlenecks for Multimodal Fusion" from NeurIPS 2021
    by Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
    https://arxiv.org/abs/2107.00135
    """

    def __init__(self,
            n_modality: int,
            bottlenecks_n: int,
            fusion_startidx: int,
            d_input: int,
            n_layers: int,
            n_head: int,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            pe_maxlen: int = 5000,
            use_pe: list = [True, True],
            mask: list = [True, True]):
        super(MBTEncoder, self).__init__()

        self.n_modality = n_modality
        self.bottlenecks_n = bottlenecks_n
        self.fusion_startIdx = fusion_startidx

        self.use_pe = use_pe
        self.mask = mask
                
        self.cls_token_per_modality = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(n_modality)])
        self.bottlenecks = nn.Parameter(torch.randn(1, self.bottlenecks_n, d_model))

        self.layer_norms_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_modality)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stacks =  nn.ModuleList(nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                num_heads = n_head,
                d_ff = d_ff,
                dropout_p = dropout
            ) for _ in range(n_modality)
        ]) for _ in range(n_layers))

    def forward(self, padded_inputs, lengths = None, return_attns = False):
        enc_slf_attn_list = []

        cls_token_per_modality = [cls_token.repeat(padded_inputs[0].size(0), 1, 1) for cls_token in self.cls_token_per_modality]
        bottlenecks = self.bottlenecks.repeat(padded_inputs[0].size(0), 1, 1)
        
        padded_inputs = [torch.cat([cls_token_per_modality[idx], padded_input], axis=1) for idx, padded_input in enumerate(padded_inputs)]
            
        self_attn_masks = []
        bottleneck_self_attn_masks = []
        if self.n_modality == 3:#####
            self.mask=[True, True, True]
        for i in range(self.n_modality):
            if self.mask[i]:
                self_attn_masks.append(get_attn_pad_mask(padded_inputs[i], lengths[i]+1, padded_inputs[i].size(1)))
            else:
                self_attn_masks.append(None)
        
        if self.n_modality == 3:#####
            self.use_pe=[True, True, True]
        enc_inputs = []
        enc_outputs = []
        for idx, pe_bool in enumerate(self.use_pe):
            if pe_bool:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](padded_inputs[idx]) +
                    self.positional_encoding(padded_inputs[idx])
                ))
            else:
                enc_outputs.append(self.dropout(
                    self.layer_norms_in[idx](padded_inputs[idx])
                ))
        
        for idx, enc_layers in enumerate(self.layer_stacks):
            enc_inputs = list(enc_outputs)
            enc_outputs = list()
            if idx < self.fusion_startIdx:
                for modal_idx, enc_layer in enumerate(enc_layers):
                    enc_output, _ = enc_layer(enc_inputs[modal_idx], self_attn_masks[modal_idx])
                    enc_outputs.append(enc_output)      
                    
            else:
                bottleneck_outputs = []
                for modal_idx, enc_layer in enumerate(enc_layers):
                    b_enc_output = torch.cat([enc_inputs[modal_idx], bottlenecks], axis=1)
                    if len(bottleneck_self_attn_masks) < self.n_modality:
                        if self.mask[i]:
                            bottleneck_self_attn_masks.append(get_attn_pad_mask(b_enc_output, lengths[modal_idx]+1+self.bottlenecks_n, b_enc_output.size(1)))
                        else:
                            bottleneck_self_attn_masks.append(None)
                            
                    enc_output, _ = enc_layer(b_enc_output, bottleneck_self_attn_masks[modal_idx])
                    
                    bottleneck_outputs.append(enc_output[:, enc_inputs[modal_idx].size(1):, :])
                    enc_output = enc_output[:, :enc_inputs[modal_idx].size(1), :]
                    enc_outputs.append(enc_output)
                    
                bottleneck_outputs_stack = torch.stack(bottleneck_outputs)
                bottlenecks = torch.mean(bottleneck_outputs_stack, dim=0)
                
        return enc_outputs, 0

class LearnableAttentionWeighting(nn.Module):
    def __init__(self):
        super(LearnableAttentionWeighting, self).__init__()
        
        self.w1 = nn.Parameter(torch.tensor(1.0))  
        self.w2 = nn.Parameter(torch.tensor(1.0))  
        self.tau = 1.0  

    def forward(self, output1, output2):

        weights = torch.stack([self.w1, self.w2])
        attention_scores = F.softmax(weights / self.tau, dim=0)
        # weighted fusion
        fused_output = attention_scores[0] * output1 + attention_scores[1] * output2
        return fused_output
    
class MLHC(nn.Module):
    def __init__(self, 
                 d_model,
                 output_dim, 
                 variables_num,
                 num_layers,
                 batch_size,
                 max_ehr_len,
                  num_heads=8,
                 n_modality=2,
                 bottlenecks_n=4,
                 dropout_rate=0.1):
        super().__init__()
        self.d_model=d_model
        self.n_modality=n_modality
        self.bottlenecks_n=bottlenecks_n
        self.dropout_rate=dropout_rate
        self.num_heads=num_heads
        self.output_dim=output_dim
        self.batch_size=batch_size
        self.num_layers=num_layers
        self.variables_num=variables_num
      
        self.ie_vslt = nn.Sequential(
                                        nn.Linear(1, self.d_model),
                                        nn.LayerNorm(self.d_model),
                                        nn.ReLU(inplace=True),
                    )
        self.ie_time = nn.Sequential(
                                    nn.Linear(1, self.d_model),
                                    nn.LayerNorm(self.d_model),
                                    nn.ReLU(inplace=True),
                )
        self.ie_feat = nn.Embedding(variables_num+1, self.d_model)
        
        
        
        
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=3,
            img_size=224,
            patch_size=32,
            hidden_size=self.d_model,
            num_heads=self.num_heads,
            proj_type="conv",
            
            dropout_rate=dropout_rate,
            spatial_dims=2,
            )

        # fusion part
        self.fusion_transformer = BimodalTransformerEncoder_MBT(
            
            batch_size = self.batch_size,
            n_modality = self.n_modality,
            bottlenecks_n = self.bottlenecks_n,      # https://arxiv.org/pdf/2107.00135.pdf # according to section 4.2 implementation details
            
            d_input = self.d_model,
            n_layers = self.num_layers,
            n_head = 8,
            d_model = self.d_model,
            d_ff = self.d_model * 4,
            dropout = dropout_rate,
            pe_maxlen = max_ehr_len+1,
            use_pe = [True, True],
            mask = [True, True],
        )


        # output
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, self.output_dim)
        )
            
        self.final_weighting=LearnableAttentionWeighting()

            
    def forward(self,ehr,ehr_mask,has_cxr, last_cxr,last_cxr_time):
        
        value_embedding = self.ie_vslt(ehr[:,:,2].unsqueeze(2))
        time_embedding = self.ie_time(ehr[:,:,0].unsqueeze(2))
        feat = ehr[:, :, 1].long()  
        feat_embedding = self.ie_feat(feat)
        # [B,L.self.d_model]
        vslt_embedding = value_embedding + time_embedding + feat_embedding 
        ehr_lengths = ehr_mask.sum(dim=1).long() 

        B,C,H,W=last_cxr.shape
        
        img_embedding = self.patch_embedding(last_cxr) # [B, patch_num,self.d_model]
        _,patch_num,_=img_embedding.shape
        cxrs_time=self.ie_time(last_cxr_time.unsqueeze(-1)) # [b,1]->[B,self.d_model]
        #[B,self.d_model]->[B,49,self.d_model]
        cxrs_time_embedding = cxrs_time.unsqueeze(1).expand(-1, patch_num, -1)
        # img type embedding
        img_feat = self.ie_feat(torch.full((B,), self.variables_num, dtype=torch.long, device=img_embedding.device))
        img_feat = img_feat.unsqueeze(1).expand(-1, patch_num, -1)  # [B, d_model] -> [B, patch_num, d_model]
        img_embedding = img_embedding + cxrs_time_embedding + img_feat 
        
        patch_num = (224 // 32) ** 2  
        cxr_lengths = torch.full((B,), patch_num, dtype=torch.long, device=img_embedding.device)
        
        outputs, _ = self.fusion_transformer(enc_outputs = [vslt_embedding, img_embedding], 
                                        fixed_lengths = [vslt_embedding.size(1), img_embedding.size(1)],
                                        varying_lengths = [ehr_lengths, cxr_lengths],
                                        fusion_idx = None,missing=has_cxr
                                        )
        
        ehr_output=outputs[0][:,0,:]
        cxr_output=outputs[1][:,0,:]
        # ehr_output=torch.nn.functional.normalize(ehr_output, dim=1)

        outputs = self.final_weighting(ehr_output, cxr_output)
        
        output = self.output_layer(outputs)
            
        return output