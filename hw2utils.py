import torch
import matplotlib
from matplotlib import pyplot as plt

class LossFunctionWithPlot:
    # A simple loss function that tracks a history each time it is called
    def __init__(self, bowl=None):
        if bowl is None:
            bowl = torch.tensor([[2.3, -0.3], [-0.5, 0.2]])
            # bowl = torch.tensor([[6.410, -0.8317], [-3.844, -0.1035]])
        self.bowl = bowl
        self.track = []
        self.losses = []
        
    def __call__(self, x):
        loss = torch.mm(self.bowl, x[:,None]).norm()
        self.track.append(x.detach().clone())
        self.losses.append(loss.detach())     
        return loss
    
    def plot_history(self, grads=None, mean_grads=None, rms_grads=None):
        # Draw the contours of the objective function, and x, and y
        if grads is not None:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 10))
        else:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5))
        for size in torch.linspace(0.01, 4.0, 41):
            angle = torch.linspace(0, 6.3, 100)
            circle = torch.stack([angle.sin(), angle.cos()])
            ellipse = torch.mm(torch.inverse(self.bowl), circle) * size
            ax1.plot(ellipse[0,:], ellipse[1,:], color='skyblue')
        track = torch.stack(self.track).t()
        ax1.set_title('progress of x')
        ax1.plot(track[0,:], track[1,:], marker='o')
        ax1.set_ylim(-1.6, 1.6)
        ax1.set_xlim(-1.6, 1.6)
        ax1.set_ylabel('x[1]')
        ax1.set_xlabel('x[0]')
        ax2.set_title('progress of loss')
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax2.plot(range(len(self.losses)), self.losses, marker='o')
        ax2.set_ylabel('loss')
        ax2.set_yscale('log', base=2)
        ax2.set_xlabel('iteration')
        if grads is not None:
            if isinstance(grads, list):
                grads = torch.stack(grads)
            ax3.set_title('progress of x[0] component of gradient')
            ax3.plot(range(len(grads)), grads[:,0], marker='o')
            ax3.set_ylabel('$\partial L/\partial x_0$')
            ax3.set_ylim(grads.abs().max() * -1.1, grads.abs().max() * 1.1)
            ax3.set_xlabel('iteration')
            if mean_grads is not None:
                ax3.axhline(mean_grads[0], label='mean', color='orange',
                        linestyle='--', linewidth=0.7)
                ax3.legend()
            if rms_grads is not None:
                ax3.axhline(rms_grads[0], label='rms', color='red', linewidth=0.5)
                ax3.legend()
            ax4.set_title('progress of x[1] component of gradient')
            ax4.plot(range(len(grads)), grads[:,1], marker='o')
            ax4.set_ylabel('$\partial L/\partial x_1$')
            ax4.set_ylim(grads.abs().max() * -1.1, grads.abs().max() * 1.1)
            ax4.set_xlabel('iteration')
            if mean_grads is not None:
                ax4.axhline(mean_grads[1], label='mean', color='orange',
                        linestyle='--', linewidth=0.7)
                ax4.legend()
            if rms_grads is not None:
                ax4.axhline(rms_grads[1], label='rms', color='red', linewidth=0.5)
                ax4.legend()
        fig.show()

class ConstantVectorNetwork(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.dimension = len(args)
        for i, a in enumerate(args):
            setattr(self, f'c{i}', torch.nn.Parameter(torch.tensor(a)))
    def forward(self):
        return torch.stack([
            getattr(self, f'c{i}')
            for i in range(self.dimension)])
