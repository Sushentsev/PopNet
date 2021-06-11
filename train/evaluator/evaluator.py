from train.loss.base import Loss


class Evaluator:
    def __init__(self, loss: Loss, batch_size: int, device):
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, test_loader):
        """
        Evaluting model on test data.
        TODO: think about batch wrapper for different models.
        :param test_loader:
        :return:
        """
        raise NotImplementedError

