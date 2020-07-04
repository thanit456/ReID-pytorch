from modeling.ft_net import ft_net, ft_net_dense, ft_net_NAS, ft_net_middle
from modeling.model import ResNet18
from modeling.pcb import PCB, PCB_test

def build_model(cfg):
    if cfg.MODEL.NAME == 'ft_net':
        model = ResNet18(cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == 'ft_net_dense':
        model = ft_net(cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == 'ft_net_dense':
        model = ft_net_dense(cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == 'ft_net_NAS':
        model = ft_net_NAS(cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == 'ft_net_middle':
        model = ft_net_middle(cfg.MODEL.NUM_CLASSES)
    
    return model 
