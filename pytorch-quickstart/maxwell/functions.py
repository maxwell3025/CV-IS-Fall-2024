import torch.utils.data
import numpy
from matplotlib import pyplot
import torchvision.utils
from datasets import load_dataset
from torch import nn
from torch import optim
from IPython import display

def handleRow(row, transform):
    row['image'] = transform(row['image'])
    return row

def getDataLoader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = load_dataset("nielsr/CelebA-faces", streaming=True, split="train") \
        .map(lambda row: handleRow(row, transform))
    return torch.utils.data.DataLoader(
        dataset, batch_size=128, num_workers=2,
    )

def displayDatasetSample(dataloader: torch.utils.data.DataLoader, device: torch.device):
    example = next(iter(dataloader))
    pyplot.figure(figsize=(8, 8))
    pyplot.axis("off")
    pyplot.title("Sample")
    pyplot.imshow(numpy.transpose(
        torchvision.utils.make_grid(
            example[0].to(device)[:64],
            padding=2,
            normalize=True
        ).cpu(),
        (1, 2, 0)
    ))
    pyplot.show()

def initializeWeights(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_size: int, num_features: int, num_channels: int, device: torch.device):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, input: numpy.ndarray):
        return self.layers(input)

class Discriminator(nn.Module):
    def __init__(self, num_channels: int, num_features: int, device: torch.device):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.to(device)
    
    def forward(self, input: numpy.ndarray):
        return self.layers(input)

def train():
    device = torch.device("mps")

    dataloader = getDataLoader()
            
    generator_network = Generator(100, 64, 3, device)

    discriminator_network = Discriminator(3, 64, device)

    criterion = nn.BCELoss()

    # This is the constant noise we use to visualize
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    REAL = 1
    FAKE = 0

    discriminator_optimizer = optim.Adam(discriminator_network.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator_network.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(1):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            #region Just grab some data and make sure the discriminator knows its fr
            discriminator_network.zero_grad()
            real_data = data["image"].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), REAL, dtype=torch.float, device=device)
            output = discriminator_network(real_data).view(-1)
            discriminator_error_real = criterion(output, label)
            discriminator_error_real.backward()
            discriminator_real_avg = output.mean().item()
            #endregion

            #region Generate fake data
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = generator_network(noise)
            #endregion

            #region Test the discriminator
            label.fill_(FAKE)
            output = discriminator_network(fake.detach()).view(-1)
            discriminator_error_fake = criterion(output, label)
            discriminator_error_fake.backward()
            discriminator_fake_avg = output.mean().item()
            discriminator_error = discriminator_error_real + discriminator_error_fake
            discriminator_optimizer.step()
            #endregion

            #region test the generator
            generator_network.zero_grad()
            label.fill_(REAL)
            output = discriminator_network(fake).view(-1)
            generator_error = criterion(output, label)
            generator_error.backward()
            generator_avg = output.mean().item()
            generator_optimizer.step()
            #endregion

            if(i % 10 != 0):
                continue

            #Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = generator_network(fixed_noise).detach().cpu()
                composite_image = numpy.transpose(
                    torchvision.utils.make_grid(
                        fake.to(device)[:64],
                        padding=2,
                        normalize=True
                    ).cpu(),
                    (1, 2, 0)
                )
                display.clear_output(wait=True)
                pyplot.imshow(composite_image)
                pyplot.show()

            print('[%d/%d][%d/%s]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (
                    epoch, 1,
                    i, "infinity?",
                    discriminator_error.item(),
                    generator_error.item(),
                    discriminator_real_avg,
                    discriminator_fake_avg, generator_avg
                )
            )
            display.update_display()
