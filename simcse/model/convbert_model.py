import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers.models.convbert.modeling_convbert import(
    ConvBertPreTrainedModel,
    ConvBertModel,
    ConvBertForMaskedLM,
)

from .init import (
                cl_init,
                corinfomax_init,
                vicreg_init,
                barlow_init,
            )

from .convbert_forwards import(
                    cl_forward,
                    corinfomax_forward,
                    vicreg_forward,
                    barlow_forward,
                    sentemb_forward,
                    normalized_sentemb_forward,
)

    

class ConvbertForCL(ConvBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.convbert = ConvBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = ConvBertForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        cl_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                
            )
        else:
            return cl_forward(
                self,
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )

class ConvbertForCorInfoMax(ConvBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.convbert = ConvBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = ConvBertForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                
            )
        else:
            return corinfomax_forward(
                self,
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )


class ConvbertForBarlow(ConvBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = ConvBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = ConvBertForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            )
        else:
            return barlow_forward(
                self,
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )

class ConvbertForVICReg(ConvBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = ConvBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = ConvBertForMaskedLM(config)
        sizes = [2048] + list(map(int, training_args.proj_output_dim.split('-')))
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        barlow_init(self,config,self.training_args)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            )
        else:
            return vicreg_forward(
                self,
                self.convbert,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )