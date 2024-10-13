import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# RBF and func approx
'''
MANUSCRIPT README!!!
left    : click to add a new node
reset   : literally
plot    : train
sinx    : approx 'sinx' (actually it's not sinx, but u can change to whatever u like in the code)
axes in right_bottom: change of loss
'''

class RBF(nn.Module):
    def __init__(self, num=20):
        super().__init__()
        self.num_node = num
        # input: x: scalar
        # weights: input->hidden
        self.a = nn.Parameter(torch.ones(self.num_node), requires_grad=True)
        # bias: gaussian func
        self.b = nn.Parameter(torch.ones(self.num_node), requires_grad=True)
        # weights: hidden->output
        self.w = nn.Parameter(torch.ones(self.num_node), requires_grad=True)
        self.init()
    def init(self):
        self.a.data.normal_(0, 0.2)
        self.b.data.normal_(0, 0.2)
        self.w.data.normal_(0, 0.2)
    def kernel_func(self, x):
        return torch.exp(-x*x*0.5)
    def forward(self, x):
        kernel_g = self.kernel_func(self.a*x+self.b)
        y = kernel_g * self.w
        return y.sum()
class RBF_Training:
    def __init__(self, num, ax_loss):
        self.model = None
        self.ax_loss = ax_loss
        self.loss_func = nn.MSELoss(reduction='sum')
        self.num = num
        self.lr = 1e-3
        self.batch_size = 1
        self.epochs = epoches
        self.optimizer = None
        self.datas = []
        self.loss = []
        self.x_plot = np.arange(5000)
    def plot_loss(self, epoch):
        size = self.loss.__len__()
        ax_loss.cla()
        ax_loss.set_xlim([max(epoch-1000, 0), epoch])
        ax_loss.set_ylim([0, 1])
        ax_loss.plot(10*self.x_plot[0:size], np.array(self.loss))
        ax_loss.figure.canvas.draw()
    def train(self):
        self.model = RBF(self.num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.init()
        self.model.train()
        data_size = self.datas.__len__()
        #for epoch in range(self.epochs):
        epoch=0
        while True:
            epoch+=1
            total_loss = 0

            for i in range(data_size):
                xx = torch.tensor([self.datas[i][0]])
                yy = torch.tensor([self.datas[i][1]])
                self.optimizer.zero_grad()
                predict = self.model(xx)
                loss = self.loss_func(predict, yy)
                loss_ = loss.item()
                total_loss += loss_

                self.lr = 5e-2if loss_>2 else 5e-3 if loss_>1.0 else 1e-3 if loss_>0.015 else 3e-5
                loss.backward()
                self.optimizer.step()
            total_loss /= data_size

            if epoch % 10 == 0:
                print(f"[Epoch {epoch+1}/{self.epochs}]: loss = {total_loss}")
                self.loss.append(total_loss)
            if total_loss < epis:
                break
        return epoch
    def test(self, x):
        return self.model(torch.tensor([x])).item()
    def input_datas(self, x, y):
        for i in range(x.__len__()):
            self.datas.append([x[i], y[i]])
class MouseClick:
    def __init__(self, ax, ax_loss):
        self.axes = ax
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes_loss = ax_loss
        self.axes_loss.set_xlim([0,epoches])
        self.axes_loss.set_ylim([0, 1])
        self.x = []
        self.y = []
        self.x_plot = np.linspace(-5, 25, 300)
        self.node_num = node_num
        self.rbf = RBF_Training(self.node_num, ax_loss)
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self)
    def __call__(self, event):
        if event.inaxes != self.axes: return
        print(f"click {event.xdata} {event.ydata}")
        self.x.append(event.xdata)
        self.y.append(event.ydata)
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes.scatter(self.x, self.y)
        self.axes.figure.canvas.draw()
    def callback_button_plot(self, event):
        self.train_model()
    def callback_button_reset(self, event):
        self.x.clear()
        self.y.clear()
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes.figure.canvas.draw()
        self.axes_loss.cla()
        self.axes_loss.set_xlim([0, epoches])
        self.axes_loss.set_ylim([0, 1])
        self.rbf.loss.clear()
        self.axes_loss.figure.canvas.draw()

    def train_model(self):
        if self.x.__len__() < 2:
            print('plz at least input 2 points!')
            return
        self.rbf.input_datas(self.x, self.y)
        epoch = self.rbf.train()
        y_plot = np.zeros(300)
        for i in range(300):
            y_plot[i] = self.rbf.test(self.x_plot[i])
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        x = np.array(self.x)
        y = np.array(self.y)
        ax.scatter(x, y)
        ax.plot(self.x_plot, y_plot)
        self.axes.figure.canvas.draw()
        self.rbf.plot_loss(epoch)
    def callback_button_sin(self, event):
        self.x.clear()
        self.y.clear()
        self.x = self.x_plot[::2]
        self.y = (2 * np.sin(0.5*self.x) + 1)
        self.rbf.input_datas(self.x, self.y)
        epoch = self.rbf.train()
        y_plot = np.zeros(300)
        for i in range(300):
            y_plot[i] = self.rbf.test(self.x_plot[i])
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        ax.plot(self.x_plot, y_plot)
        ax.plot(self.x, self.y)
        self.axes.figure.canvas.draw()
        self.rbf.plot_loss(epoch)

epoches = 1000
node_num = 64
epis = 0.002

fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.45, right=0.95, top=0.95)
ax.set_xlim([-5, 25])
ax.set_ylim([-3, 8])

ax_loss = fig.add_axes([0.3, 0.05,0.65, 0.35])

trainRbf = MouseClick(ax, ax_loss)

ax_reset = fig.add_axes([0.05, 0.85, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(trainRbf.callback_button_reset)

ax_plot = fig.add_axes([0.05, 0.70, 0.2, 0.1])
button_plot = widgets.Button(ax_plot, 'plot')
button_plot.on_clicked(trainRbf.callback_button_plot)

ax_sin = fig.add_axes([0.05, 0.55, 0.2, 0.1])
button_sin = widgets.Button(ax_sin, 'sinx')
button_sin.on_clicked(trainRbf.callback_button_sin)

print(torch.cuda.is_available())
plt.show()
