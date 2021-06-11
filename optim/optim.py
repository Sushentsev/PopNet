class Optimizer:
    def __init__(self, optim, scheduler=None, max_grad_norm=0.):
        self.optim = optim
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm


