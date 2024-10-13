import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from functools import partial
import math

# 曲线细分
'''
MANUSCRIPT README!!!
left    : click to add a new node
reset   : literally
checkBoard : choose a method u like
generationSlider: sigma in gaussian basis
alphaSlider: coefficient of 4 points method
'''
class subdivision:
    def __init__(self, axes: plt.axes):
        self.x = []
        self.y = []
        self.sub_x = []
        self.sub_y = []
        self.method = [self.method_chaikin, self.method_cubic_B_spline, self.method_four_points]
        self.generation = 0
        self.axes = axes
        self.alpha = 0.1
        self.old_line = None
        self.old_subdivision = [None, None, None]
        self.visible = [False, False, False]
        self.num = 0
        self.closed = False
    def add_node(self, x, y):
        self.num += 1
        self.x.append(x)
        self.y.append(y)
        if self.num > 1:
            self.old_line.remove()
        self.old_line, = self.axes.plot(self.x, self.y, linestyle='--', marker='o', color='green')
        self.update()
        self.axes.figure.canvas.draw()
    def update(self):
        for i in range(3):
            if self.visible[i]:
                self.method[i]()
            else:
                self.clear_subdivision(i)
    def clear_all(self):
        self.x.clear()
        self.y.clear()
        self.sub_x.clear()
        self.sub_y.clear()
        self.num = 0
        for i in range(3):
            if self.old_subdivision[i] is not None:
                self.clear_subdivision(i)
        if self.old_line is not None:
            self.old_line.remove()
            self.old_line = None
        self.axes.figure.canvas.draw()
    def clear_subdivision(self, index):
        if self.old_subdivision[index] is not None:
            self.old_subdivision[index].remove()
            self.old_subdivision[index] = None
        self.axes.figure.canvas.draw()
    def method_chaikin(self):
        if self.generation == 0 or self.num == 0:
            self.clear_subdivision(0)
            return
        x_temp = []
        y_temp = []
        num = self.num
        self.sub_x = self.x[:]
        self.sub_y = self.y[:]
        for i in range(0, self.generation):
            for j in range(num-1):
                x_temp.append(0.75*self.sub_x[j]+0.25*self.sub_x[j+1])
                x_temp.append(0.25*self.sub_x[j]+0.75*self.sub_x[j+1])
                y_temp.append(0.75*self.sub_y[j]+0.25*self.sub_y[j+1])
                y_temp.append(0.25*self.sub_y[j]+0.75*self.sub_y[j+1])
            if self.closed:
                x_temp.append(0.75 * self.sub_x[num-1] + 0.25 * self.sub_x[0])
                x_temp.append(0.25 * self.sub_x[num-1] + 0.75 * self.sub_x[0])
                y_temp.append(0.75 * self.sub_y[num-1] + 0.25 * self.sub_y[0])
                y_temp.append(0.25 * self.sub_y[num-1] + 0.75 * self.sub_y[0])
                num *=2
            else:
                num = 2*num-2
            self.sub_x = x_temp[:]
            self.sub_y = y_temp[:]
            x_temp.clear()
            y_temp.clear()
        self.clear_subdivision(0)
        if self.closed:
            self.sub_x.append(self.sub_x[0])
            self.sub_y.append(self.sub_y[0])
        self.old_subdivision[0], = self.axes.plot(self.sub_x, self.sub_y)
        self.axes.figure.canvas.draw()
    def method_cubic_B_spline(self):
        if self.generation == 0 or self.num == 0 or self.num == 1:
            self.clear_subdivision(1)
            return
        x_temp = []
        y_temp = []
        num = self.num
        self.sub_x = self.x[:]
        self.sub_y = self.y[:]
        for i in range(0, self.generation):
            if self.closed:
                x_temp.append(0.125*(self.sub_x[num-1]+self.sub_x[1])+0.75*self.sub_x[0])
                x_temp.append(0.5*(self.sub_x[0]+self.sub_x[1]))
                y_temp.append(0.125 * (self.sub_y[num - 1] + self.sub_y[1]) + 0.75 * self.sub_y[0])
                y_temp.append(0.5 * (self.sub_y[0] + self.sub_y[1]))
            else:   # 不封闭时直接保留首尾点，不做任何处理
                x_temp.append(self.sub_x[0])
                x_temp.append(0.5 * (self.sub_x[0] + self.sub_x[1]))
                y_temp.append(self.sub_y[0])
                y_temp.append(0.5 * (self.sub_y[0] + self.sub_y[1]))
            for j in range(1, num-1):
                x_temp.append(0.125 * (self.sub_x[j-1] + self.sub_x[j+1]) + 0.75 * self.sub_x[j])
                x_temp.append(0.5 * (self.sub_x[j] + self.sub_x[j+1]))
                y_temp.append(0.125 * (self.sub_y[j-1] + self.sub_y[j+1]) + 0.75 * self.sub_y[j])
                y_temp.append(0.5 * (self.sub_y[j] + self.sub_y[j+1]))
            if self.closed:
                x_temp.append(0.125 * (self.sub_x[num-2] + self.sub_x[0]) + 0.75 * self.sub_x[num-1])
                x_temp.append(0.5 * (self.sub_x[num-1] + self.sub_x[0]))
                y_temp.append(0.125 * (self.sub_y[num-2] + self.sub_y[0]) + 0.75 * self.sub_y[num-1])
                y_temp.append(0.5 * (self.sub_y[num-1] + self.sub_y[0]))
                num *=2
            else:
                x_temp.append(self.sub_x[num-1])
                y_temp.append(self.sub_y[num-1])
                num = 2*num-1
            self.sub_x = x_temp[:]
            self.sub_y = y_temp[:]
            x_temp.clear()
            y_temp.clear()
        self.clear_subdivision(1)
        if self.closed:
            self.sub_x.append(self.sub_x[0])
            self.sub_y.append(self.sub_y[0])
        self.old_subdivision[1], = self.axes.plot(self.sub_x, self.sub_y)
        self.axes.figure.canvas.draw()
    def method_four_points(self):
        if self.generation == 0 or self.num<4:
            self.clear_subdivision(2)
            return
        x_temp = []
        y_temp = []
        num = self.num
        self.sub_x = self.x[:]
        self.sub_y = self.y[:]
        for i in range(0, self.generation):
            x_temp.append(self.sub_x[0])
            x_temp.append(0.5 * (self.sub_x[0] + self.sub_x[1])+0.5*self.alpha*
                          (self.sub_x[0]+self.sub_x[1]-self.sub_x[num-1]-self.sub_x[2]))
            y_temp.append(self.sub_y[0])
            y_temp.append(0.5 * (self.sub_y[0] + self.sub_y[1]) + 0.5 * self.alpha *
                          (self.sub_y[0] + self.sub_y[1] - self.sub_y[num-1] - self.sub_y[2]))
            for j in range(1, num - 2):
                x_temp.append(self.sub_x[j])
                x_temp.append(0.5 * (self.sub_x[j] + self.sub_x[j+1]) + 0.5 * self.alpha *
                              (self.sub_x[j] + self.sub_x[j+1] - self.sub_x[j-1] - self.sub_x[j+2]))
                y_temp.append(self.sub_y[j])
                y_temp.append(0.5 * (self.sub_y[j] + self.sub_y[j+1]) + 0.5 * self.alpha *
                              (self.sub_y[j] + self.sub_y[j+1] - self.sub_y[j-1] - self.sub_y[j+2]))
            x_temp.append(self.sub_x[num - 2])
            x_temp.append(0.5 * (self.sub_x[num - 2] + self.sub_x[num - 1]) + 0.5 * self.alpha *
                          (self.sub_x[num - 2] + self.sub_x[num - 1] - self.sub_x[num - 3] - self.sub_x[0]))
            y_temp.append(self.sub_y[num - 2])
            y_temp.append(0.5 * (self.sub_y[num - 2] + self.sub_y[num - 1]) + 0.5 * self.alpha *
                          (self.sub_y[num - 2] + self.sub_y[num - 1] - self.sub_y[num -3] - self.sub_y[0]))
            x_temp.append(self.sub_x[num-1])
            x_temp.append(0.5 * (self.sub_x[0] + self.sub_x[num-1]) + 0.5 * self.alpha *
                          (self.sub_x[0] + self.sub_x[num-1] - self.sub_x[num-2] - self.sub_x[1]))
            y_temp.append(self.sub_y[num-1])
            y_temp.append(0.5 * (self.sub_y[0] + self.sub_y[num-1]) + 0.5 * self.alpha *
                          (self.sub_y[0] + self.sub_y[num-1] - self.sub_y[num-2] - self.sub_y[1]))
            num *= 2
            self.sub_x = x_temp[:]
            self.sub_y = y_temp[:]
            x_temp.clear()
            y_temp.clear()
        self.clear_subdivision(2)
        self.sub_x.append(self.sub_x[0])
        self.sub_y.append(self.sub_y[0])
        self.old_subdivision[2], = self.axes.plot(self.sub_x, self.sub_y)
        self.axes.figure.canvas.draw()
    def check_board(self, index):
        if self.old_subdivision[index] is not None:
            self.clear_subdivision(index)
        else:
            self.method[index]()
class MouseClick:
    def __init__(self, axes):
        self.axes = axes
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.subdivision = subdivision(self.axes)
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.axes: return
        print('click', event.xdata, event.ydata)
        self.subdivision.add_node(event.xdata, event.ydata)
    def callback_button_check(self, event):
        index = check_label.index(event)
        self.subdivision.visible[index] = not self.subdivision.visible[index]
        self.subdivision.update()

    def callback_button_plot(self, event):
        self.plot_result(self.axes)
    def callback_button_reset(self, event):
        self.subdivision.clear_all()
    def callback_slider_gene(self, event):
        self.subdivision.generation = event
        self.subdivision.update()
    def callback_button_check_closed(self, event):
        self.subdivision.closed = not self.subdivision.closed
        self.subdivision.update()
    def callback_slider_alpha(self, event):
        self.subdivision.alpha = event
        self.subdivision.update()


fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

plotter = MouseClick(ax)

ax_reset = fig.add_axes([0.05, 0.80, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(plotter.callback_button_reset)

ax_check = fig.add_axes([0.05, 0.4, 0.2, 0.2])
check_label = ['chaikin', 'cubic_B_spline', '4points_subdivision']
button_check = widgets.CheckButtons(ax_check, labels=check_label)
button_check.on_clicked(plotter.callback_button_check)

ax_check_closed = fig.add_axes([0.05, 0.3, 0.2, 0.05])
button_check_closed = widgets.CheckButtons(ax_check_closed, labels=['closed'])
button_check_closed.on_clicked(plotter.callback_button_check_closed)

ax_slider_gene = fig.add_axes([0.05, 0.2, 0.2, 0.05])
slider_gene = widgets.Slider(ax_slider_gene, 'generation', 0, 8, valinit=0, valstep=1)
slider_gene.on_changed(plotter.callback_slider_gene)

ax_slider_alpha = fig.add_axes([0.05, 0.1, 0.2, 0.05])
slider_alpha = widgets.Slider(ax_slider_alpha, 'slider', 0, 0.6, valinit=0.1)
slider_alpha.on_changed(plotter.callback_slider_alpha)

plt.show()