import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from functools import partial
import math

# 三次样条函数 全局
'''
MANUSCRIPT README!!!
left    : click to add a new node(the sequence of node.x should be increasing)
right   : drag a node arbitrarily in its original support
          (of course u can drag it anywhere u like unless ur looking for a wrong plot)
reset   : literally

WHY u can edit its tangent vec -> 'cause i'm lazy :)
should parameterize and handle x, y separately
'''

class spline:
    def __init__(self, ax):
        self.x = []
        self.y = []
        self.num = 0
        self.x_plot = np.linspace(0, 1.0, 21)
        self.axes = ax
        self.node = None
    def natural_spline(self):
        x = np.array(self.x)
        y = np.array(self.y)
        h = x[1:] - x[0:-1] # 0-n-1
        b = ((y[2:] - y[1:-1])/h[1:] - (y[1:-1] - y[0:-2])/h[0:-1])*6
        b = np.hstack([0, b, 0])
        diag = 2*(h[0:-1]+h[1:])
        diag = np.hstack([1, diag, 1])
        A = np.diag(diag,k=0) + np.diag(np.hstack([h[0:-1], 0]),k=-1) + np.diag(np.hstack([0, h[1:]]), k=1)
        m = np.linalg.solve(A, b)
        cubic = np.hstack([
            y[0:-1].reshape((-1, 1)),
            ((y[1:] - y[0:-1])/h[0:]-h[0:]/6*(m[1:self.num]+2*m[0:self.num-1])).reshape((-1, 1)),
            (m[0:self.num-1]/2).reshape((-1, 1)),
            ((m[1:self.num]-m[0:self.num-1])/6/h[0:]).reshape((-1, 1))
        ])
        self.plot_spline(cubic)
    def clear_ax_and_scatter(self):
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.node = self.axes.plot(np.array(self.x), np.array(self.y), 'o', picker=True, pickradius=5)
        self.axes.figure.canvas.draw()
    def plot_spline(self, cubic):
        x = []
        y = []
        self.clear_ax_and_scatter()
        for i in range(self.num-1):
            x_plot = np.linspace(self.x[i], self.x[i+1], 21)
            for j in range(x_plot.size):
                x.append(x_plot[j])
                y.append((array_powers(x_plot[j]-self.x[i],3)*cubic[i]).sum())
        self.axes.plot(x, y)
        self.axes.figure.canvas.draw()
    def append_and_plot(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.num += 1
        if self.num >= 2:
            self.natural_spline()
        else:
            self.clear_ax_and_scatter()

def array_powers(x, power: int):
    a = [1.0]
    for i in range(1, power + 1):
        a.append(x ** i)
    return np.array(a)


class MouseClick:
    def __init__(self, ax):
        self.axes = ax
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.spline = spline(self.axes)
        self.pick_ind = -1
        self.cid_press = self.axes.figure.canvas.mpl_connect('button_press_event', self)
        self.cid_pick = self.axes.figure.canvas.mpl_connect('pick_event', self.pick_node)
        self.cid_move = self.axes.figure.canvas.mpl_connect('motion_notify_event', self.move_node)
        self.cid_release = self.axes.figure.canvas.mpl_connect('button_release_event', self.release_node)

    def __call__(self, event):
        if event.button != 1: return
        if event.inaxes != self.axes:
            return
        print('click', event.xdata, event.ydata)
        self.spline.append_and_plot(event.xdata, event.ydata)
    def callback_button_reset(self, event):
        self.spline.x.clear()
        self.spline.y.clear()
        self.spline.num = 0
        self.spline.clear_ax_and_scatter()
    def pick_node(self, event):
        if event.mouseevent.button != 3: return
        artist = event.artist
        xdata = artist.get_xdata()
        ydata = artist.get_ydata()
        self.pick_ind = event.ind[0]
        print(f"onpick point: ({xdata[self.pick_ind]}, {ydata[self.pick_ind]}), ind: {self.pick_ind}")
    def move_node(self, event):
        if event.button != 3 or self.pick_ind < 0 or event.inaxes != self.axes: return
        # print(self.pick_ind)
        self.spline.x[self.pick_ind] = event.xdata
        self.spline.y[self.pick_ind] = event.ydata
        # self.spline.node[0].set_xdata(self.spline.x) # ??? what is Line2d
        # self.spline.node[0].set_ydata(self.spline.y)
        self.spline.natural_spline()
        self.axes.figure.canvas.draw()
    def release_node(self, event):
        self.pick_ind = -1

fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

plotter = MouseClick(ax)

ax_reset = fig.add_axes([0.05, 0.80, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(plotter.callback_button_reset)

plt.show()