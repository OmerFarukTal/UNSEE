import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers.models.deberta_v2.modeling_deberta_v2 import(
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    DebertaV2ForMaskedLM,
)

from .init import (
                cl_init,
                corinfomax_init,
                vicreg_init,
                barlow_init,
            )

from .deberta_forwards import(
                    cl_forward,
                    corinfomax_forward,
                    vicreg_forward,
                    barlow_forward,
                    sentemb_forward,
                    normalized_sentemb_forward,
)

    

class DebertaForCL(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = DebertaV2Model(config)

     

        if self.model_args.do_mlm:
            self.lm_head = DebertaV2ForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        cl_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return normalized_sentemb_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                
            )
        else:
            return cl_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )

class DebertaForCorInfoMax(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = DebertaV2Model(config)

     

        if self.model_args.do_mlm:
            self.lm_head = DebertaV2ForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return normalized_sentemb_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                
            )
        else:
            return corinfomax_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )


class DebertaForBarlow(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = DebertaV2Model(config)

     

        if self.model_args.do_mlm:
            self.lm_head = DebertaV2ForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return normalized_sentemb_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return barlow_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )

class DebertaForVICReg(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = DebertaV2Model(config)

     

        if self.model_args.do_mlm:
            self.lm_head = DebertaV2ForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return normalized_sentemb_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return vicreg_forward(
                self,
                self.deberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )