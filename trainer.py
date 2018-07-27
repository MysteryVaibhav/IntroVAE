import sys
import torch
import numpy as np
from model import IntroVAE
from torch.nn import functional as F
from timeit import default_timer as timer
from torchvision.utils import save_image


class Trainer:
    def __init__(self, params, data_loader, evaluator, util):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.util = util

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train(self):
        model = IntroVAE(self.params, self.util)
        if self.params.cuda:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.wdecay)
        num_of_mini_batches = len(self.data_loader.train_data_loader.dataset) // self.params.batch_size
        prev_best = float('inf')
        self.evaluator.test_loss(model, self.loss_function, 0)
        try:
            for epoch in range(self.params.num_epochs):
                losses = []
                start_time = timer()
                for iters, (data, _) in enumerate(self.data_loader.train_data_loader):
                    model.train()
                    optimizer.zero_grad()
                    data = self.util.to_variable(data)
                    recon_batch, mu, logvar = model(data)
                    loss = self.loss_function(recon_batch, data, mu, logvar)
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())
                    avg_loss = np.asscalar(np.mean(losses))
                    optimizer.step()
                    sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
                                     iters, num_of_mini_batches, avg_loss))
                    sys.stdout.flush()
                print("Epoch {} : Training Loss: {:.5f}, Time elapsed {:.2f} mins".format(epoch + 1, avg_loss,
                      (timer() - start_time) / 60))
                if (epoch + 1) % self.params.validate_every == 0:
                    test_loss = self.evaluator.test_loss(model, self.loss_function, epoch + 1)
                    if test_loss < prev_best:
                        prev_best = test_loss
                        print("Test loss reduced, saving model !!")
                        torch.save(model.state_dict(), self.params.model_dir + 'best_model_weights.t7')

                    # Sampling few random samples from z to see the quality
                    sample = self.util.to_variable(
                        torch.randn(self.params.batch_size, self.params.latent_dimension))
                    sample = model.decode(sample).cpu()
                    save_image(sample.view(64, 1, 28, 28).data,
                               self.params.result_dir + 'sample_' + str(epoch + 1) + '.png')

        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(model.state_dict(), self.params.model_dir + 'model_weights_interrupt.t7')