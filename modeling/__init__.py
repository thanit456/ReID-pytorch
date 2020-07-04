# from .model import ft_net

def build_model(cfg):
    model = ft_net(cfg.MODEL.NUM_CLASSES)
    return model 