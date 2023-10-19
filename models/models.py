
def create_model(opt):
    model = None
    from .sagan import SelfAttentionGANModel
    model = SelfAttentionGANModel()
    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
