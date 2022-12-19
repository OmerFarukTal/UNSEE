import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers.models.squeezebert.modeling_squeezebert import(
    SqueezeBertPreTrainedModel,
    SqueezeBertModel,
    SqueezeBertForMaskedLM,
)

from .init import (
                cl_init,
                corinfomax_init,
                vicreg_init,
                barlow_init,
            )

from .squeeze_bert_forwards import(
                    cl_forward,
                    corinfomax_forward,
                    vicreg_forward,
                    barlow_forward,
                    sentemb_forward,
                    normalized_sentemb_forward,
)

    

class SqueezebertForCL(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.squeezebert = SqueezeBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = SqueezeBertForMaskedLM(config)
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
                self.squeezebert,
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
                self.squeezebert,
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

class SqueezebertForCorInfoMax(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.squeezebert = SqueezeBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = SquuezeBertForMaskedLM(config)
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
                self.squeezebert,
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
                self.squeezebert,
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


class SqueezebertForBarlow(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = SqueezeBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = SqueezeBertForMaskedLM(config)
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
                self.squeezebert,
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
                self.squeezebert,
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

class SqueezebertForVICReg(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, training_args, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.training_args = training_args
        self.deberta = SqueezeBertModel(config)

     

        if self.model_args.do_mlm:
            self.lm_head = SqueezeBertForMaskedLM(config)
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
                self.squeezebert,
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
                self.squeezebert,
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