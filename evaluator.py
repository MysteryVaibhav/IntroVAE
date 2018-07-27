import torch
from torchvision.utils import save_image


class Evaluator:
    def __init__(self, params, data_loader, util):
        self.params = params
        self.data_loader = data_loader
        self.util = util

    def test_loss(self, model, loss_function, epoch):
        model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.data_loader.test_data_loader):
            data = self.util.to_variable(data)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data.cpu().numpy()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(self.params.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu().data,
                           self.params.result_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.data_loader.test_data_loader.dataset)
        return test_loss