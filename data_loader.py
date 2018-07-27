import torch
from torchvision import transforms, datasets


class Fashion(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.urls = [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        ]
        super(Fashion, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                      download=download)


class DataLoader:
    def __init__(self, params, util):
        self.params = params
        self.util = util
        self.train_data = Fashion('data', train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        self.test_data = Fashion('data', train=False, download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.params.cuda else {}
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.params.batch_size,
                                                             shuffle=True, **kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.params.batch_size,
                                                            shuffle=False, **kwargs)
