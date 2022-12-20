import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


from transformers.modeling_outputs import SequenceClassifierOutput,BaseModelOutputWithPoolingAndCrossAttentions

from ..loss import CovarianceLoss, invariance_loss
from ..save_utils import SaveResults


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1))
    )  # (bs, num_sent, hidden)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [
                [0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1))
                + [0.0] * i
                + [z3_weight]
                + [0.0] * (z1_z3_cos.size(-1) - i - 1)
                for i in range(z1_z3_cos.size(-1))
            ]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
)

def corinfomax_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    cls_output = outputs.last_hidden_state[:, 0].detach().cpu().numpy()

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1))
    )  # (bs, num_sent, hidden)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    cl_loss = loss_fct(cos_sim, labels)
    
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)

    cov_loss = cls.cov_loss_fct(z1, z2)
    sim_loss = cls.inv_loss_fct(z1, z2)
    cos_loss = -torch.trace(cos_sim)*cls.training_args.cos_loss_weight/cls.training_args.per_device_train_batch_size
    loss = 0
    
    if "cov" in cls.model_args.loss_elements.split("-"):
        loss += cov_loss * cls.training_args.cov_loss_weight 

    if "inv" in cls.model_args.loss_elements.split("-"):
        loss += sim_loss * cls.training_args.sim_loss_weight
    
    if "cos" in cls.model_args.loss_elements.split("-"):
        loss += cos_loss

    if "entropy" in cls.model_args.loss_elements.split("-"):
        loss += cl_loss

    save_results = cls.save_results
    writer = cls.writer 

    save_results.track_values(loss.item(), cov_loss.item(), sim_loss.item(), cl_loss.item(), cos_loss.item(), cls_output)
    save_results.save_values(writer, cls.cov_loss_fct, cls.training_args.cov_loss_weight, cls.training_args.sim_loss_weight, encoder.encoder.layer[-1].output.LayerNorm.weight, encoder.encoder.layer[-1].output.LayerNorm.bias)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def vicreg_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    cls_output = outputs.last_hidden_state[:, 0].detach().cpu().numpy()

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1))
    )  # (bs, num_sent, hidden)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    
    repr_loss = F.mse_loss(z1, z2)
    z1 = z1 -z1.mean(dim=0)
    z2 = z2 -z2.mean(dim = 0)
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim = 0)+ 0.0001)
    std_loss = torch.mean(F.relu(1-std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) /2
    cov_z1 = (z1.T @ z1) / (cls.training_args.batch_size -1)
    cov_z2 = (z2.T@ z2) /(cls.training_args.batch_size -1)

    cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(8192) + off_diagonal(cov_z2).pow_(2).sum().div(8192)

    loss = (cls.training_args.sim_coeff * repr_loss
            + cls.training_args.std_coeff * std_loss
            + cls.training_args.cov_coeff * cov_loss)
    save_results = cls.save_results
    writer = cls.writer 

    save_results.track_values(loss.item(), cov_loss.item(), repr_loss.item(), std_loss.item(), cls_output)
    save_results.save_values(writer, cls.training_args.cov_coeff, cls.training_args.sim_coeff, cls.training_args.std_coeff, encoder.encoder.layer[-1].output.LayerNorm.weight, encoder.encoder.layer[-1].output.LayerNorm.bias)
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
def barlow_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    cls_output = outputs.last_hidden_state[:, 0].detach().cpu().numpy()

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1))
    )  # (bs, num_sent, hidden)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    c = cls.bn(z1).T @ cls.bn(z2)
    c.div_(cls.training_args.batch_size)
    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=c_list, tensor = c.contiguous())
        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        c_list[dist.get_rank()] = c
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        c = torch.sum(c,0)

    
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    
 

    #c = cls.bn(z1).T @ cls.bn(z2)

    # sum the cross-correlation matrix between all gpus
    #c.div_(cls.training_args.batch_size)
    #torch.distributed.all_reduce(c)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + cls.training_args.lambd * off_diag

    
    save_results = cls.save_results
    writer = cls.writer 

    #save_results.track_values(loss.item(), cov_loss.item(), repr_loss.item(), std_loss.item(), 0, cls_output)
    #save_results.save_values(writer, cls.training_args.std_coeff, cls.training_args.cov_coeff, cls.training_args.sim_coeff, encoder.encoder.layer[-1].output.LayerNorm.weight, encoder.encoder.layer[-1].output.LayerNorm.bias)
    save_results.track_values(loss.item(),on_diag.item(),off_diag.item(),cls_output)
    save_results.save_values(writer,cls.training_args.lambd,encoder.encoder.layer[-1].output.LayerNorm.weight,encoder.encoder.layer[-1].output.LayerNorm.bias)
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )



def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

def normalized_sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    pooler_output = F.normalize(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )