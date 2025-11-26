# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
代码摘要生成模型 - Seq2Seq with CodeBERT Encoder
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class Seq2Seq(nn.Module):
    """
    Encoder-Decoder 模型用于代码摘要生成
    Encoder: CodeBERT
    Decoder: Transformer Decoder
    """
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None,
                 sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def _tie_or_clone_weights(self, first_module, second_module):
        """将 lm_head 权重与 decoder embedding 绑定"""
        first_module.weight = second_module.weight
    
    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.encoder.embeddings.word_embeddings)
    
    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None):
        # 编码器
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0]  # [batch_size, source_len, hidden_size]
        
        if target_ids is not None:
            # 训练模式
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids)
            
            # 解码器
            out = self.decoder(
                tgt_embeddings,
                encoder_output,
                tgt_mask=attn_mask,
                memory_key_padding_mask=~source_mask.bool()
            )
            hidden_states = torch.tanh(self.dense(out))
            lm_logits = self.lm_head(hidden_states)
            
            # 计算损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return loss, loss * target_mask[:, 1:].sum(), target_mask[:, 1:].sum()
        else:
            # 推理模式 - beam search
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0) if source_ids.is_cuda else torch.LongTensor(1).fill_(0)
            
            for i in range(source_ids.shape[0]):
                context = encoder_output[i:i+1, :, :]
                context_mask = source_mask[i:i+1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                
                if source_ids.is_cuda:
                    input_ids = input_ids.cuda()
                
                context = context.repeat(self.beam_size, 1, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids)
                    
                    out = self.decoder(
                        tgt_embeddings,
                        context,
                        tgt_mask=attn_mask,
                        memory_key_padding_mask=~context_mask.bool()
                    )
                    hidden_states = torch.tanh(self.dense(out[:, -1, :]))
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(
                        input_ids.data.index_select(0, beam.getCurrentOrigin())
                    )
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1)
                        for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))
            
            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    """Beam Search 实现"""
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        self.sos = sos
        self.eos = eos
        self.scores = torch.FloatTensor(size).zero_()
        self.allScores = []
        self.prevKs = []
        self.nextYs = [torch.LongTensor(size).fill_(self.sos)]
        self.finished = []
    
    def getCurrentState(self):
        return self.nextYs[-1]
    
    def getCurrentOrigin(self):
        return self.prevKs[-1]
    
    def advance(self, wordLk):
        numWords = wordLk.size(1)
        
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self.eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        
        self.allScores.append(self.scores)
        self.scores = bestScores
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self.eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))
        
        if self.nextYs[-1][0] == self.eos:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        
        return True
    
    def done(self):
        return len(self.finished) >= self.size
    
    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        return self.finished[0]
    
    def getHyp(self, beam_res):
        hyp = []
        for j in range(len(self.prevKs[:beam_res[1]]) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][beam_res[2]])
            beam_res = (beam_res[0], beam_res[1], self.prevKs[j][beam_res[2]])
        return hyp[::-1]
    
    def buildTargetTokens(self, hyp):
        tokens = []
        for i in range(len(hyp)):
            tokens.append(hyp[i])
            if hyp[i] == self.eos:
                break
        return tokens
