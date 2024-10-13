import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from functools import partial

# 函数插值和拟合
'''
MANUSCRIPT README!!!
left    : click to add a new node
reset   : literally
plot    : literally(if u have plotted and u add a new node, 'plot' again to update
checkBoard : choose a method u like
sigmaSlider: sigma in gaussian basis
power      : determine the basis {1,x...x^power} in LS and Ridge
lambda     : in Ridge
'''
def array_powers(x, power: int):
    a = [1.0]
    for i in range(1, power + 1):
        a.append(x ** i)
    return np.array(a)
class polynomial_interp:
    def __init__(self):
        self.name = 'polynomial'
        self.x = np.array(0)
        self.y = np.array(0)
        self.a = np.array(0)
        self.plot = True
        self.visible = False
    def calc_a(self):
        num = self.x.size
        x = self.x.reshape((num, 1))
        y = self.y.reshape((num, 1))
        b = np.ones((num, 1))
        for i in range(1, num):
            b = np.hstack((b, x ** i))
        self.a = (np.linalg.inv(b) @ y).flatten()
    def set_nodes(self, x, y):
        self.x = x
        self.y = y
        self.calc_a()
    def plot_result(self, x, ax):
        if not self.plot:
            return
        y_plot = np.zeros(x.size)
        for i in range(x.size):
            y_plot[i] = (array_powers(x[i], self.a.size - 1) * self.a).sum()
        ax.plot(x, y_plot, label='polynomial')
    def change_visibility(self):
        self.visible = (not self.visible)
        print('polynomial visibility: ' + str(not self.visible) + '->' + str(self.visible))

class gaussian_interp:
    def __init__(self):
        self.name = 'gaussian'
        self.x = np.array(0)
        self.y = np.array(0)
        self.a = np.array(0)
        self.sigma2 = 25.0
        self.plot = True
        self.visible = False
    def calc_gaussian(self, x, xi):
        return np.exp(-(x - xi)**2/2.0/self.sigma2)
    def calc_a(self):
        num = self.x.size
        x = np.vstack((self.x.reshape((num, 1)), (self.x[-2] + self.x[-1])/2))
        y = np.vstack((self.y.reshape((num, 1)), (self.y[-2] + self.y[-1])/2))
        num += 1
        b = np.ones((num, 1))
        for i in range(num-1):
            arr = []
            for j in range(num):
                arr.append(self.calc_gaussian(x[j], x[i]))
            b = np.hstack((b, np.array(arr).reshape((num, 1))))
        self.a = (np.linalg.inv(b) @ y).flatten()
    def set_nodes(self, x, y):
        self.x = x
        self.y = y
        self.calc_a()
    def plot_result(self, x, ax):
        if not self.plot:
            return
        y_plot = np.zeros(x.size)
        for i in range(x.size):
            arr = [1.0]
            for j in range(self.x.size):
                arr.append(self.calc_gaussian(x[i], self.x[j]))
            y_plot[i] = (np.array(arr) * self.a).sum()
        ax.plot(x, y_plot, label='gaussian')
    def change_visibility(self):
        self.visible = (not self.visible)
        print('gaussian visibility: ' + str(not self.visible) + '->' + str(self.visible))
class leastSquare_approx:
    def __init__(self, ridge: bool = False):
        self.name = 'leastSquare'if not ridge else 'ridge'
        self.x = np.array(0)
        self.y = np.array(0)
        self.a = np.array(0)
        self.power = 3
        self.ridge = ridge
        self.lamda = 2
        self.plot = True
        self.visible = False
    def calc_a(self):
        if self.power>self.x.size-1:
            self.power = self.x.size - 1
            print('Warning '+('[ridge]'if self.ridge else'[leastSquare]:')
                  +': power > point num, power is clamped to ', self.power)
        num = self.x.size
        b = np.zeros((1, self.power+1))
        for i in range(self.x.size):
            b = np.vstack((b, array_powers(self.x[i], self.power)))
        b = b[1:, ]
        lamda = self.lamda if self.ridge else 0
        self.a = np.linalg.inv(b.T@b+lamda*np.eye(self.power+1))@b.T@self.y
    def set_nodes(self, x, y):
        self.x = x
        self.y = y
        if self.power>self.x.size-1:
            self.power = self.x.size - 1
            print('Warning ' + ('[ridge]' if self.ridge else '[leastSquare]')
                  + ': power > point num, power is clamped to ', self.power)
        self.calc_a()
    def plot_result(self, x, ax):
        if not self.plot:
            return
        y_plot = np.zeros(x.size)
        for i in range(x.size):
            y_plot[i] = (array_powers(x[i], self.power)*self.a).sum()
        ax.plot(x, y_plot, label=('Ridge'if self.ridge else 'leastSquare'))
    def change_visibility(self):
        self.visible = (not self.visible)
        print(('ridge visibility: 'if self.ridge else'leastSquare visibility: ')
            + str(not self.visible) + '->' + str(self.visible))
class MouseClick:
    def __init__(self, ax):
        self.axes = ax
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.func = []
        self.x = []
        self.y = []
        self.x_plot = np.linspace(-5, 25, 300)
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self)
    def plot_result(self, ax):
        if self.x.__len__() < 2:
            print('plz at least input 2 points!')
            return
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        x = np.array(self.x)
        y = np.array(self.y)
        ax.scatter(x, y)
        for i in self.func:
            if i.visible:
                i.set_nodes(x, y)
                i.plot_result(self.x_plot, ax)
        self.axes.legend()
        self.axes.figure.canvas.draw()
    def add_func(self, func):
        self.func.append(func)
    def __call__(self, event):
        if event.inaxes != self.axes: return
        print('click', event.xdata, event.ydata)
        self.x.append(event.xdata)
        self.y.append(event.ydata)
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes.scatter(self.x, self.y)
        self.axes.figure.canvas.draw()
    def callback_button_check(self, event):
        index = check_label.index(event)
        self.func[index].change_visibility()
    def callback_button_plot(self, event):
        self.plot_result(self.axes)
    def callback_button_reset(self, event):
        self.x.clear()
        self.y.clear()
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes.figure.canvas.draw()
    def callback_button_selectAll(self, check_button, event):
        not_select = False
        for i in self.func:
            if not i.visible:
                not_select = True
                check_button.set_active(check_label.index(i.name))
        if not not_select:
            for i in self.func:
                check_button.set_active(check_label.index(i.name))
        check_button.ax.figure.canvas.draw()
    def callback_slider_sigma(self, event):
        self.func[1].sigma2 = event**2
    def callback_slider_lamda(self, event):
        self.func[3].lamda = event
    def callback_slider_power(self, event):
        self.func[2].power = self.func[3].power = event
        if self.func[2].x == []:
            self.func[2].calc_a()
            self.func[3].calc_a()




fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

plotter = MouseClick(ax)
plotter.add_func(polynomial_interp())
plotter.add_func(gaussian_interp())
plotter.add_func(leastSquare_approx())
plotter.add_func(leastSquare_approx(ridge=True))

ax_reset = fig.add_axes([0.05, 0.80, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(plotter.callback_button_reset)

ax_plot = fig.add_axes([0.05, 0.65, 0.2, 0.1])
button_plot = widgets.Button(ax_plot, 'plot')
button_plot.on_clicked(plotter.callback_button_plot)

ax_check = fig.add_axes([0.05, 0.4, 0.2, 0.2])
check_label = ['polynomial', 'gaussian', 'leastSquare', 'ridge']
button_check = widgets.CheckButtons(ax_check, labels=check_label)
button_check.on_clicked(plotter.callback_button_check)

ax_selectAll = fig.add_axes([0.05, 0.3, 0.2, 0.1])
button_selectAll = widgets.Button(ax_selectAll, label='Select all')
button_selectAll.on_clicked(partial(plotter.callback_button_selectAll, button_check))

ax_slider_sigma = fig.add_axes([0.05, 0.2, 0.2, 0.05])
slider_sigma = widgets.Slider(ax_slider_sigma, 'sigma', 0.1, 10, valinit=5)
slider_sigma.on_changed(plotter.callback_slider_sigma)

ax_slider_power = fig.add_axes([0.05, 0.15, 0.2, 0.05])
slider_power = widgets.Slider(ax_slider_power, 'power', 0, 12, valinit=3, valstep=1)
slider_power.on_changed(plotter.callback_slider_power)

ax_slider_lamda = fig.add_axes([0.05, 0.10, 0.2, 0.05])
slider_lamda = widgets.Slider(ax_slider_lamda, 'lamda', 0.0, 50, valinit=2)
slider_lamda.on_changed(plotter.callback_slider_lamda)

plt.show()