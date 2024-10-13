import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from functools import partial
import math

# 分段Bezier
# it is better to store vec but not point in bezier.control_point, i am lazy to fix my code
'''
MANUSCRIPT README!!!
left    : add a new node(in a sequence) (bezier curve appears only when u click at least 4 nodes)
middle  : hit a node to show its tangent vec(once again to hide it) or move tangent vec arbitrarily
right   : move existed nodes or scale tangent vec of a node along its original direction
reset   : clear all
'''

class bezier:
    def __init__(self, ax):
        self.x = []
        self.y = []
        self.control_point = [] # 存2，3两个点
        self.num = 0
        self.x_plot = np.linspace(0, 1.0, 21)
        self.axes = ax
        self.t_pace = np.linspace(0, 1, 21)
        self.tangent = []
        self.tangent_show = []
    def init_bezier(self): # 当有4个型值点时才plot
        vec01 = np.array([self.x[1]-self.x[0], self.y[1]-self.y[0]])
        vec02 = np.array([self.x[2]-self.x[0], self.y[2]-self.y[0]])
        vec32 = np.array([self.x[2]-self.x[3], self.y[2]-self.y[3]])
        vec31 = np.array([self.x[1]-self.x[3], self.y[1]-self.y[3]])
        x12 = vec02[0]/6+self.x[1]
        y12 = vec02[1]/6+self.y[1]
        x03 = -vec02[0]/6+self.x[1]
        y03 = -vec02[1]/6+self.y[1]
        x13 = vec31[0]/6+self.x[2]
        y13 = vec31[1]/6+self.y[2]
        x22 = -vec31[0]/6+self.x[2]
        y22 = -vec31[1]/6+self.y[2]
        val0 = (vec01/np.linalg.norm(vec01)*vec02/6).sum()/np.linalg.norm(vec01)
        p02 = np.array([x03, y03])-vec01*(1-2*val0)
        val3 = (vec32/np.linalg.norm(vec32)*vec31/6).sum()/np.linalg.norm(vec32)
        p23 = np.array([x22, y22])-vec32 * (1 - 2 * val3)
        self.control_point.append([p02[0], p02[1], x03, y03])
        self.control_point.append([x12, y12, x13, y13])
        self.control_point.append([x22, y22, p23[0], p23[1]])
        self.plot_all_curve()

    def move_node(self, ind):
        if ind == 0:
            vec02 = np.array([self.x[2]-self.x[0], self.y[2]-self.y[0]])
            vec01 = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
            x12 = vec02[0] / 6 + self.x[1]
            y12 = vec02[1] / 6 + self.y[1]
            x03 = -vec02[0] / 6 + self.x[1]
            y03 = -vec02[1] / 6 + self.y[1]
            val0 = (vec01 / np.linalg.norm(vec01) * vec02 / 6).sum() / np.linalg.norm(vec01)
            p02 = np.array([x03, y03]) - vec01 * (1 - 2 * val0)
            self.control_point[0][0] = p02[0]
            self.control_point[0][1] = p02[1]
            self.control_point[0][2] = x03
            self.control_point[0][3] = y03
            self.control_point[1][0] = x12
            self.control_point[1][1] = y12
        elif ind == 1:
            vec = [0, 0, 0]
            for i in range(1, 3):
                vec[i] = np.array([self.x[ind + i] - self.x[ind - 2 + i], self.y[ind + i] - self.y[ind - 2 + i]]) / 6
            for i in range(1, 3):
                self.control_point[ind - 2 + i][2] = self.x[ind - 1 + i] - vec[i][0]
                self.control_point[ind - 2 + i][3] = self.y[ind - 1 + i] - vec[i][1]
                self.control_point[ind - 1 + i][0] = self.x[ind - 1 + i] + vec[i][0]
                self.control_point[ind - 1 + i][1] = self.y[ind - 1 + i] + vec[i][1]
        elif ind == self.num-2:
            vec = [0, 0, 0]
            for i in range(0, 2):
                vec[i] = np.array([self.x[ind + i] - self.x[ind - 2 + i], self.y[ind + i] - self.y[ind - 2 + i]]) / 6
            for i in range(0, 2):
                self.control_point[ind - 2 + i][2] = self.x[ind - 1 + i] - vec[i][0]
                self.control_point[ind - 2 + i][3] = self.y[ind - 1 + i] - vec[i][1]
                self.control_point[ind - 1 + i][0] = self.x[ind - 1 + i] + vec[i][0]
                self.control_point[ind - 1 + i][1] = self.y[ind - 1 + i] + vec[i][1]
        elif ind == self.num-1:
            self.control_point.pop()
            self.add_new()
        else:
            vec = [0, 0, 0]
            for i in range(3):
                vec[i] = np.array([self.x[ind+i] - self.x[ind - 2+i], self.y[ind+i] - self.y[ind - 2+i]]) / 6
            for i in range(3):
                self.control_point[ind - 2+i][2] = self.x[ind - 1+i] - vec[i][0]
                self.control_point[ind - 2+i][3] = self.y[ind - 1+i] - vec[i][1]
                self.control_point[ind - 1+i][0] = self.x[ind - 1+i] + vec[i][0]
                self.control_point[ind - 1+i][1] = self.y[ind - 1+i] + vec[i][1]
        self.plot_all_curve()
    def clear_ax_and_scatter(self):
        self.axes.cla()
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.node = self.axes.plot(np.array(self.x), np.array(self.y), 'o', picker=True, pickradius=5)
        self.axes.figure.canvas.draw()
    def plot_spline(self, x1, y1, x2, y2, x3, y3, x4, y4):
        x = []
        y = []
        for t in self.t_pace:
            x.append((1-t)**3*x1+3*t*(1-t)**2*x2+3*t**2*(1-t)*x3+t**3*x4)
            y.append((1-t)**3*y1+3*t*(1-t)**2*y2+3*t**2*(1-t)*y3+t**3*y4)

        self.axes.plot(x, y)
        self.axes.figure.canvas.draw()
    def add_new(self):
        ind = self.num-2
        vec = np.array([self.x[ind-1]-self.x[ind+1], self.y[ind-1]-self.y[ind+1]])
        x_3 = vec[0]/6+self.x[ind]
        y_3 = vec[1]/6+self.y[ind]
        x__2 = -vec[0]/6+self.x[ind]
        y__2 = -vec[1]/6+self.y[ind]
        vec1 = np.array([self.x[ind]-self.x[ind+1], self.y[ind]-self.y[ind+1]])
        val = (vec1 / np.linalg.norm(vec1) * vec / 6).sum() / np.linalg.norm(vec1)
        p__3 = np.array([x__2, y__2])-vec1*(1-2*val)
        self.control_point[ind-1][2] = x_3
        self.control_point[ind-1][3] = y_3
        self.control_point.append([x__2, y__2, p__3[0], p__3[1]])
        self.plot_all_curve()
    def plot_all_curve(self):
        for i in range(self.num-1):
            self.plot_spline(self.x[i],self.y[i],self.control_point[i][0],
                             self.control_point[i][1],self.control_point[i][2],
                             self.control_point[i][3],self.x[i+1],self.y[i+1])
    def append_and_plot(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.tangent.append(None)
        self.clear_ax_and_scatter()
        self.num += 1
        if self.num == 4:
            self.init_bezier()
        elif self.num > 4:
            self.add_new()
    def plot_tangent(self, ind, forcetodraw = False):
        if self.tangent[ind] != None and forcetodraw == False:
            self.tangent[ind].remove()
            self.axes.figure.canvas.draw()
            self.tangent[ind] = None
            return
        x = []
        y = []
        if ind == 0:
            x = [self.x[0], self.control_point[0][0]]
            y = [self.y[0], self.control_point[0][1]]

        elif ind == self.num-1:
            x = [self.x[ind], self.control_point[ind-1][2]]
            y = [self.y[ind], self.control_point[ind-1][3]]
        else:
            x = [self.control_point[ind - 1][2], self.x[ind], self.control_point[ind][0]]
            y = [self.control_point[ind - 1][3], self.y[ind], self.control_point[ind][1]]
        points, = self.axes.plot(x, y, 'o-', picker=True, pickradius=5)
        self.axes.figure.canvas.draw()
        self.tangent[ind] = points

class MouseClick:
    def __init__(self, ax):
        self.axes = ax
        self.axes.set_xlim([-5, 25])
        self.axes.set_ylim([-3, 8])
        self.bezier = bezier(self.axes)
        self.pick_ind = -1
        self.move_type = -1
        self.tang_ind = -1
        self.cid_press = self.axes.figure.canvas.mpl_connect('button_press_event', self)
        self.cid_pick = self.axes.figure.canvas.mpl_connect('pick_event', self.pick_node)
        self.cid_move = self.axes.figure.canvas.mpl_connect('motion_notify_event', self.move_node)
        self.cid_release = self.axes.figure.canvas.mpl_connect('button_release_event', self.release_node)

    def __call__(self, event):
        if event.button != 1: return
        if event.inaxes != self.axes:
            return
        print('click', event.xdata, event.ydata)
        self.bezier.append_and_plot(event.xdata, event.ydata)
    def callback_button_reset(self, event):
        self.bezier.x.clear()
        self.bezier.y.clear()
        self.bezier.tangent.clear()
        self.bezier.control_point.clear()
        self.bezier.num = 0
        self.bezier.clear_ax_and_scatter()
    def pick_node(self, event):
        if event.mouseevent.button == 1: return
        artist = event.artist
        xdata = artist.get_xdata()
        ydata = artist.get_ydata()
        num = xdata.__len__()
        self.pick_ind = event.ind[0]
        if event.mouseevent.button == 2 and num>3:
            self.bezier.plot_tangent(self.pick_ind)
            return
        if event.mouseevent.button == 3 and num<=3:
            print(f"onpick point: ({xdata[self.pick_ind]}, {ydata[self.pick_ind]}), ind: {self.pick_ind}")
            self.tang_ind = self.bezier.tangent.index(artist)
            self.move_type = 1
            return
        if event.mouseevent.button == 3 and num>3:
            print(f"onpick point: ({xdata[self.pick_ind]}, {ydata[self.pick_ind]}), ind: {self.pick_ind}")
            self.move_type = 2
            return
        if event.mouseevent.button == 2 and num<=3:
            print(f"onpick point: ({xdata[self.pick_ind]}, {ydata[self.pick_ind]}), ind: {self.pick_ind}")
            self.tang_ind = self.bezier.tangent.index(artist)
            self.move_type = 3
    def move_node(self, event):
        if event.button == 1 or self.pick_ind < 0 or event.inaxes != self.axes: return
        if self.move_type == 2:
            # print(self.pick_ind)
            self.bezier.x[self.pick_ind] = event.xdata
            self.bezier.y[self.pick_ind] = event.ydata
            # self.spline.node[0].set_xdata(self.spline.x) # ??? what is Line2d
            # self.spline.node[0].set_ydata(self.spline.y)
            self.bezier.clear_ax_and_scatter()
            self.bezier.move_node(self.pick_ind)
            self.axes.figure.canvas.draw()
            return
        x = []
        y = []
        if self.pick_ind == 2:
            if self.move_type == 1:
                node = np.array([self.bezier.x[self.tang_ind], self.bezier.y[self.tang_ind]])
                unit_vec = np.array([self.bezier.control_point[self.tang_ind][0]-node[0],
                                     self.bezier.control_point[self.tang_ind][1]-node[1]])
                cur_vec = np.array([event.xdata, event.ydata]) - node
                cur_pos = (cur_vec*unit_vec).sum()/(np.linalg.norm(unit_vec)**2)*unit_vec + node
                self.bezier.control_point[self.tang_ind][0] = cur_pos[0]
                self.bezier.control_point[self.tang_ind][1] = cur_pos[1]
            else:
                self.bezier.control_point[self.tang_ind][0] = event.xdata
                self.bezier.control_point[self.tang_ind][1] = event.ydata
            x = [self.bezier.control_point[self.tang_ind-1][2],
                 self.bezier.x[self.tang_ind],
                 self.bezier.control_point[self.tang_ind][0]]
            y = [self.bezier.control_point[self.tang_ind - 1][3],
                 self.bezier.y[self.tang_ind],
                 self.bezier.control_point[self.tang_ind][1]]
        elif self.pick_ind == 0:
            if self.move_type == 1:
                node = np.array([self.bezier.x[self.tang_ind], self.bezier.y[self.tang_ind]])
                unit_vec = np.array([self.bezier.control_point[self.tang_ind-1][2] - node[0],
                                     self.bezier.control_point[self.tang_ind-1][3] - node[1]])
                cur_vec = np.array([event.xdata, event.ydata]) - node
                cur_pos = (cur_vec * unit_vec).sum() / (np.linalg.norm(unit_vec)**2) * unit_vec + node
                self.bezier.control_point[self.tang_ind-1][2] = cur_pos[0]
                self.bezier.control_point[self.tang_ind-1][3] = cur_pos[1]
            else:
                self.bezier.control_point[self.tang_ind-1][2] = event.xdata
                self.bezier.control_point[self.tang_ind-1][3] = event.ydata
            x = [self.bezier.control_point[self.tang_ind-1][2],
                 self.bezier.x[self.tang_ind],
                 self.bezier.control_point[self.tang_ind][0]
                 ]
            y = [self.bezier.control_point[self.tang_ind-1][3],
                 self.bezier.y[self.tang_ind],
                 self.bezier.control_point[self.tang_ind][1],
                 ]
        else:
            if self.tang_ind == 0:
                self.bezier.control_point[0][0] = event.xdata
                self.bezier.control_point[0][1] = event.ydata
                x = [self.bezier.x[0], event.xdata]
                y = [self.bezier.y[0], event.ydata]
            else:
                ind = self.bezier.num-2
                self.bezier.control_point[ind][2] = event.xdata
                self.bezier.control_point[ind][3] = event.ydata
                x = [self.bezier.x[ind+1], event.xdata]
                y = [self.bezier.y[ind+1], event.ydata]
        self.bezier.clear_ax_and_scatter()
        self.bezier.plot_all_curve()
        for i, iter in enumerate(self.bezier.tangent):
            if iter != None:
                self.bezier.plot_tangent(i, True)
        # print(self.bezier.tangent[self.tang_ind].get_visible())
        self.axes.figure.canvas.draw()


    def release_node(self, event):
        self.pick_ind = -1
        self.move_type = -1

fig, ax = plt.subplots( figsize=(10, 6))
plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

plotter = MouseClick(ax)

ax_reset = fig.add_axes([0.05, 0.80, 0.2, 0.1])
button_reset = widgets.Button(ax_reset, 'reset')
button_reset.on_clicked(plotter.callback_button_reset)

plt.show()