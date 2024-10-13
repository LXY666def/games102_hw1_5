import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from functools import partial
import math

# 参数化、
'''
left    : click to add a new node
reset   : literally
replot  : literally
checkBoard  : choose a param method u like
limit to max power : max power = numOfNode - 1
powerSlider : literally
'''

def array_powers(x, power: int):
    a = [1.0]
    for i in range(1, power + 1):
        a.append(x ** i)
    return np.array(a)
class leastSquare_approx:
    def __init__(self, ridge: bool = False):
        self.name = 'leastSquare'if not ridge else 'ridge'
        self.x = np.array(0)
        self.y = np.array(0)
        self.a = np.array(0)
        self.power = 4
        self.actual_power = self.power
        self.ridge = ridge
        self.lamda = 2
        self.plot = True
        self.limit2max_power = False
    def calc_a(self):
        self.actual_power = self.power
        if self.actual_power > self.x.size-1 or self.limit2max_power:
            self.actual_power = self.x.size - 1
            print('Warning '+('[ridge]'if self.ridge else'[leastSquare]')
                  +': power > point num, power is clamped to ', self.actual_power)
        num = self.x.size
        b = np.zeros((1, self.actual_power+1))
        for i in range(self.x.size):
            b = np.vstack((b, array_powers(self.x[i], self.actual_power)))
        b = b[1:, ]
        lamda = self.lamda if self.ridge else 0
        self.a = np.linalg.inv(b.T@b+lamda*np.eye(self.actual_power+1))@b.T@self.y
    def set_nodes(self, x, y):
        self.x = x
        self.y = y
        self.calc_a()
        return self
    def plot_result(self, x, ax):
        if not self.plot:
            return
        y_plot = np.zeros(x.size)
        for i in range(x.size):
            y_plot[i] = (array_powers(x[i], self.actual_power)*self.a).sum()
        ax.plot(x, y_plot, label=('Ridge'if self.ridge else 'leastSquare'))
    def approx(self, x):
        if not self.plot:
            return
        y_plot = np.zeros(x.size)
        for i in range(x.size):
            y_plot[i] = (array_powers(x[i], self.actual_power)*self.a).sum()
        return y_plot
class Sequence:
    def __init__(self, ax):
        self.axes = ax
        self.x = []
        self.y = []
        self.param_method = [self.parameter_equal, self.parameter_chordal
                , self.parameter_centripetal, self.parameter_foley_neilson]
        self.ls_method = [leastSquare_approx(), leastSquare_approx(True)]
        self.x_plot = np.linspace(0, 1.0, 101)
        self.map_param_method = {
            'equal': 0, 'chordal': 1, 'centripetal': 2, 'foley_neilson': 3
        }
        self.visible = [False, False, False, False]
    def parameter_equal(self):
        data_num = self.x.__len__()
        param_dist = np.linspace(0,1.0, data_num)
        self.approx_plot(param_dist, 'equal')
    def parameter_chordal(self):
        data_num = self.x.__len__()
        param_dist = [0.0]
        total = 0
        for i in range(1, data_num):
            dist = math.sqrt((self.x[i] - self.x[i - 1]) ** 2 + (self.y[i] - self.y[i - 1]) ** 2)
            param_dist.append(param_dist[i - 1] + dist)
            total += dist
        self.approx_plot(np.array(param_dist) / total, 'chordal')
    def parameter_centripetal(self):
        data_num = self.x.__len__()
        param_dist = [0.0]
        total = 0
        for i in range(1, data_num):
            dist = math.sqrt(math.sqrt((self.x[i] - self.x[i - 1]) ** 2 + (self.y[i] - self.y[i - 1]) ** 2))
            param_dist.append(param_dist[i-1]+dist)
            total += dist
        self.approx_plot(np.array(param_dist)/total, 'centripetal')
    def parameter_foley_neilson(self):
        data_num = self.x.__len__()
        if data_num == 2:
            self.approx_plot(np.array([0, 1]), 'foley_neilson')
            return
        norm2_dist = np.zeros(data_num)
        for i in range(1, data_num):
            norm2_dist[i] = (self.x[i]-self.x[i-1])**2+(self.y[i]-self.y[i-1])**2
        norm1_dist = np.sqrt(norm2_dist)
        alpha_hat = np.zeros(data_num)
        for i in range(1, data_num-1):
            dist2 = (self.x[i+1]-self.x[i-1])**2+(self.y[i+1]-self.y[i-1])**2
            alpha_hat[i] = min(
                np.pi-np.arccos((norm2_dist[i+1]+norm2_dist[i]-dist2)/norm1_dist[i]/norm1_dist[i+1]*0.5),
                np.pi*0.5
            )
        total = norm1_dist[1]*(1+1.5*alpha_hat[1]*norm1_dist[1]/(norm1_dist[1]+norm1_dist[2]))
        param_dist = [0, total]
        for i in range(2, data_num-1):
            temp = norm1_dist[i]*(1+1.5*alpha_hat[i-1]*norm1_dist[i-1]/(norm1_dist[i-1]+norm1_dist[i])
                                  + 1.5*alpha_hat[i]*norm1_dist[i]/(norm1_dist[i]+norm1_dist[i+1]))
            param_dist.append(temp + param_dist[-1])
            total += temp
        temp = norm1_dist[-1]*(1+1.5*alpha_hat[-2]*norm1_dist[-2]/(norm1_dist[-2]+norm1_dist[-1]))
        param_dist.append(temp + param_dist[-1])
        total += temp
        self.approx_plot(np.array(param_dist) / total, 'foley_neilson')
    def clear_ax_and_scatter(self):
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.axes.scatter(np.array(self.x), np.array(self.y))
        self.axes.figure.canvas.draw()
    def approx_plot(self, param_dist, label_):
        xx = self.ls_method[0].set_nodes(param_dist, np.array(self.x)).approx(self.x_plot)
        yy = self.ls_method[0].set_nodes(param_dist, np.array(self.y)).approx(self.x_plot)
        self.axes.plot(xx, yy, label=label_)
        self.axes.legend()
        self.axes.figure.canvas.draw()
    def append_point(self, x=None, y=None):
        if x != None:
            self.x.append(x)
            self.y.append(y)
        self.clear_ax_and_scatter()
        if self.x.__len__() < 2:
            return
        for ind, method in enumerate(self.param_method):
            if self.visible[ind]:
                method()

class MouseClick:
    def __init__(self, ax):
        self.axes = ax
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.sequence = Sequence(self.axes)
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.axes: return
        print('click', event.xdata, event.ydata)
        self.sequence.append_point(event.xdata, event.ydata)
    def callback_button_reset(self, event):
        self.sequence.x.clear()
        self.sequence.y.clear()
        self.sequence.clear_ax_and_scatter()
    def callback_button_check(self, event):
        index = self.sequence.map_param_method[event]
        self.sequence.visible[index] = not self.sequence.visible[index]
    def callback_button_limit(self, event):
        self.sequence.ls_method[0].limit2max_power = not self.sequence.ls_method[0].limit2max_power
    def callback_button_replot(self, event):
        self.sequence.append_point()
    def callback_slider_power(self, event):
        self.sequence.ls_method[0].power = event


fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

plotter = MouseClick(ax)

ax_reset = fig.add_axes([0.05, 0.80, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(plotter.callback_button_reset)

ax_replot = fig.add_axes([0.05, 0.65, 0.2, 0.1])
button_replot = widgets.Button(ax_replot, 'replot')
button_replot.on_clicked(plotter.callback_button_replot)

ax_check = fig.add_axes([0.05, 0.40, 0.2, 0.2])
check_label = ['equal', 'chordal', 'centripetal', 'foley_neilson']
button_check = widgets.CheckButtons(ax_check, labels=check_label)
button_check.on_clicked(plotter.callback_button_check)

ax_limit = fig.add_axes([0.05, 0.3, 0.2, 0.05])
button_limit = widgets.CheckButtons(ax_limit, labels=['limit to max power'])
button_limit.on_clicked(plotter.callback_button_limit)

ax_slider_power = fig.add_axes([0.05, 0.20, 0.2, 0.05])
slider_power = widgets.Slider(ax_slider_power, 'power', 0, 10, valinit=4, valstep=1)
slider_power.on_changed(plotter.callback_slider_power)

plt.show()