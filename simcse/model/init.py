from ..loss import CovarianceLoss, invariance_loss
from .utils import MLPLayer,OneLayeredMLPLayer,Similarity,Pooler
from ..save_utils import SaveResults,SaveResultsVICReg,SaveResultsBarlow

def cl_init(cls, config, training_args):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config, training_args)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def corinfomax_init(cls, config, training_args):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if (cls.model_args.pooler_type == "cls") and (training_args.do_proj):
        cls.mlp = MLPLayer(config, training_args)
    elif (cls.model_args.pooler_type == "cls"):
        cls.mlp = OneLayeredMLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.cov_loss_fct = CovarianceLoss(cls.training_args)
    cls.inv_loss_fct = invariance_loss
    cls.init_weights()

    cls.save_results = SaveResults(cls.model_args, cls.training_args)
    cls.writer = SaveResults.create_results(cls.save_results)
def vicreg_init(cls, config, training_args):
    """
    VICReg learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config, training_args)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    cls.save_results = SaveResultsVICReg(cls.model_args, cls.training_args)
    cls.writer = SaveResultsVICReg.create_results(cls.save_results)

def barlow_init(cls, config, training_args):
    """
    Barlow learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config, training_args)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    cls.save_results = SaveResultsBarlow(cls.model_args, cls.training_args)
    cls.writer = SaveResultsBarlow.create_results(cls.save_results)