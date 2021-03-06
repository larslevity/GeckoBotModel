# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:25:25 2019

@author: ls
"""
import numpy as np
import matplotlib.pyplot as plt

from Src.Utils import save


def draw_point_dir(point, direction, size=12, label=None, colp='red'):
    x1, y1 = point
    dx, dy = direction
    plt.plot([x1], [y1], marker='o', color=colp, markersize=size)
    plt.plot([x1, x1+dx], [y1, y1+dy], color='blue')
    if label:
        plt.text(x1+dx, y1+dy, label)
    plt.axis('equal')


def draw_line(point1, point2, color='gray', linestyle='dashed'):
    x1, y1 = point1
    x2, y2 = point2
    plt.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle)


g = {  # manually tuned
     '000': [('100', ((.0, .0), -1)), ('010', ((0, 0), 1)),
             ('000', ((0, 0), 0))],

     '100': [  # ('000', ((0, 0), 0)),
             ('100f', ((0, .71), 1)), ('100f', ((0.15, -.2), -80)),
             ('100f', ((.1, .4), -10)), ('102', ((.1, .3), -20))],
     '100f': [('100df', None)],
     '100df': [('010', ((0, .71), 1)), ('110', ((.15, -.2), -80)),
               ('101', ((.1, .4), -10))],
     '110': [('110f', None)],
     '110f': [('110df', None)],
     '110df': [('111', None)],
     '111': [('111f', None)],
     '111f': [('111df', None)],
     '111df': [('112', None)],
     '112': [('112f', None)],
     '112f': [('112df', ((0, 1), -1)), ('110df', ((0.15, -.2), -80)),
              ('115df', ((0, 1), 1)), ('113df', ((-0.15, -.2), 80))],
     '112df': [('100', None)],
#     '104': [('104f', None)],
#     '104f': [('104df', None)],
#     '104df': [('105', None)],
#     '105': [('105f', None)],
#     '105f': [('105df', None)],
#     '105df': [('106', ((0, .2), -130)), ('107', ((0, .4), -70))],
#     '106': [('106f', None)],
#     '106f': [('106df', None)],
#     '106df': [('105', None)],
#     '107': [('107f', None)],
#     '107f': [('107df', None)],
#     '107df': [('100', None)],
     '101': [('101f', None)],
     '101f': [('101df', None)],
     '101df': [('100', None)],
     '102': [('102f', None)],
     '102f': [('102df', None)],
     '102df': [('103', None)],
     '103': [('103f', None)],
     '103f': [('103df', None)],
     '103df': [('102', ((0, .4), -30)), ('100', ((0, .6), -15))],

     '010': [  # ('000', ((0, 0), 0)),
             ('010f', ((0, .71), -1)), ('010f', ((-.15, -.2), 80)),
             ('010f', ((-.1, .4), 10)), ('012', ((-.1, .4), 20))],
     '010f': [('010df', None)],
     '010df': [('100', ((0, .71), -1)), ('113', ((-.15, -.2), 80)),
               ('011', ((-.1, .4), 10))],
     '113': [('113f', None)],
     '113f': [('113df', None)],
     '113df': [('114', None)],
     '114': [('114f', None)],
     '114f': [('114df', None)],
     '114df': [('115', None)],
     '115': [('112f', None)],
#     '115f': [('115df', ((0, 1), 1)), ('113df', ((-0.15, -.2), 80))],
     '115df': [('010', None)],
#     '014': [('014f', None)],
#     '014f': [('014df', None)],
#     '014df': [('015', None)],
#     '015': [('015f', None)],
#     '015f': [('015df', None)],
#     '015df': [('016', ((0, .2), 130)), ('017', ((0, .4), 70))],
#     '016': [('016f', None)],
#     '016f': [('016df', None)],
#     '016df': [('015', None)],
#     '017': [('017f', None)],
#     '017f': [('017df', None)],
#     '017df': [('010', None)],
     '011': [('011f', None)],
     '011f': [('011df', None)],
     '011df': [('010', None)],
     '012': [('012f', None)],
     '012f': [('012df', None)],
     '012df': [('013', None)],
     '013': [('013f', None)],
     '013f': [('013df', None)],
     '013df': [('012', ((0, .4), 30)), ('010', ((0, 0.6), 15))]
    }

ref = {
     '000': [[0, 0, 0, 0, 0], [1, 0, 0, 0], .1],

     '100': [[0, 90, 90, 0, 90], [0, 1, 1, 0], .8],
     '100f': [[0, 90, 90, 0, 90], [1, 1, 1, 1], .1],
     '100df': [[0, 90, 90, 0, 90], [1, 0, 0, 1], .1],

     '110': [[45, 45, 0, 45, 45], [1, 0, 0, 1], .8],
     '110f': [[45, 45, 0, 45, 45], [1, 1, 1, 1], .1],
     '110df': [[45, 45, 0, 45, 45], [1, 1, 0, 0], .1],

     '111': [[45, 45, -90, 45, 45], [1, 1, 0, 0], .8],
     '111f': [[45, 45, -90, 45, 45], [1, 1, 1, 1], .1],
     '111df': [[45, 45, -90, 45, 45], [0, 0, 1, 1], .1],

     '112': [[45, 45, 10, 45, 45], [0, 0, 1, 1], .8],
     '112f': [[45, 45, 0, 45, 45], [1, 1, 1, 1], .1],
     '112df': [[45, 45, 0, 45, 45], [0, 1, 1, 0], .1],

     '104': [[50, 30, 90, 30, 150], [1, 0, 0, 1], .8],
     '104f': [[50, 30, 90, 30, 150], [1, 1, 1, 1], .1],
     '104df': [[50, 30, 90, 30, 150], [0, 1, 1, 0], .1],

     '105': [[124, 164, 152, 62, 221], [0, 1, 1, 0], .8],
     '105f': [[124, 164, 152, 62, 221], [1, 1, 1, 1], .1],
     '105df': [[124, 164, 152, 62, 221], [1, 0, 0, 1], .1],

     '106': [[0, 0, 24, 0, 0], [1, 0, 0, 1], .8],
     '106f': [[0, 0, 24, 0, 0], [1, 1, 1, 1], .1],
     '106df': [[0, 0, 24, 0, 0], [0, 1, 1, 0], .1],

     '107': [[30, 90, 80, 10, 10], [1, 0, 0, 1], .8],
     '107f': [[30, 90, 80, 10, 10], [1, 1, 1, 1], .1],
     '107df': [[30, 90, 80, 10, 10], [0, 1, 1, 0], .1],

     '101': [[40, 1, -10, 60, 10], [1, 0, 0, 1], .8],
     '101f': [[40, 1, -10, 60, 10], [1, 1, 1, 1], .1],
     '101df': [[40, 1, -10, 60, 10], [0, 1, 1, 0], .1],

     '102': [[48, 104, 114, 27, 124], [0, 1, 1, 0], .8],
     '102f': [[48, 104, 114, 27, 124], [1, 1, 1, 1], .1],
     '102df': [[48, 104, 114, 27, 124], [1, 0, 0, 1], .1],

     '103': [[1, 72, -10, 1, 55], [1, 0, 0, 1], .8],
     '103f': [[1, 72, -10, 1, 55], [1, 1, 1, 1], .1],
     '103df': [[1, 72, -10, 1, 55], [0, 1, 1, 0], .1],

     #   LEFT

     '010': [[90, 0, -90, 90, 0], [1, 0, 0, 1], .8],
     '010f': [[90, 0, -90, 90, 0], [1, 1, 1, 1], .1],
     '010df': [[90, 0, -90, 90, 0], [0, 1, 1, 0], .1],

     '113': [[45, 45, 0, 45, 45], [0, 1, 1, 0], .8],
     '113f': [[45, 45, 0, 45, 45], [1, 1, 1, 1], .1],
     '113df': [[45, 45, 0, 45, 45], [1, 1, 0, 0], .1],

     '114': [[45, 45, 90, 45, 45], [1, 1, 0, 0], .8],
     '114f': [[45, 45, 90, 45, 45], [1, 1, 1, 1], .1],
     '114df': [[45, 45, 90, 45, 45], [0, 0, 1, 1], .1],

     '115': [[45, 45, -10, 45, 45], [0, 0, 1, 1], .8],
#     '115f': [[45, 45, 0, 45, 45], [1, 1, 1, 1], .1],
     '115df': [[45, 45, 0, 45, 45], [1, 0, 0, 1], .1],

     '014': [[30, 50, -90, 150, 30], [0, 1, 1, 0], .8],
     '014f': [[30, 50, -90, 150, 30], [1, 1, 1, 1], .1],
     '014df': [[30, 50, -90, 150, 30], [1, 0, 0, 1], .1],

     '015': [[164, 124, -152, 221, 62], [1, 0, 0, 1], .8],
     '015f': [[164, 124, -152, 221, 62], [1, 1, 1, 1], .1],
     '015df': [[164, 124, -152, 221, 62], [0, 1, 1, 0], .1],

     '016': [[0, 0, -24, 0, 0], [0, 1, 1, 0], .8],
     '016f': [[0, 0, -24, 0, 0], [1, 1, 1, 1], .1],
     '016df': [[0, 0, -24, 0, 0], [1, 0, 0, 1], .1],

     '017': [[90, 30, -80, 10, 10], [0, 1, 1, 0], .8],
     '017f': [[90, 30, -80, 10, 10], [1, 1, 1, 1], .1],
     '017df': [[90, 30, -80, 10, 10], [1, 0, 0, 1], .1],

     '011': [[1, 40, 10, 10, 60], [0, 1, 1, 0], .8],
     '011f': [[1, 40, 10, 10, 60], [1, 1, 1, 1], .1],
     '011df': [[1, 40, 10, 10, 60], [1, 0, 0, 1], .1],

     '012': [[104, 48, -114, 124, 27], [1, 0, 0, 1], .8],
     '012f': [[104, 48, -114, 124, 27], [1, 1, 1, 1], .1],
     '012df': [[104, 48, -114, 124, 27], [0, 1, 1, 0], .1],

     '013': [[72, 1, -70, 55, 1], [0, 1, 1, 0], .8],
     '013f': [[72, 1, -70, 55, 1], [1, 1, 1, 1], .1],
     '013df': [[72, 1, -70, 55, 1], [1, 0, 0, 1], .1]
      }


class CandidateHandler(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(CandidateHandler, self).__setitem__(key, [])
        self[key].append(value)

    def minimum(self):
        min_d = {}
        for key in self:
            min_d[key] = min([abs(c) for c in self[key]])
        min_ = min([abs(min_d[key]) for key in min_d])
        min_key = min(min_d, key=min_d.get)
        return min_, min_key

    def maximum(self):
        max_d = {}
        for key in self:
            max_d[key] = max([abs(c) for c in self[key]])
        max_ = min([abs(max_d[key]) for key in max_d])
        max_key = min(max_d, key=max_d.get)
        return max_, max_key


class ReferenceGenerator(object):
    def __init__(self, pose='000', graph=g, ref=ref):
        self.graph = Graph(graph)
        self.pose = pose
        self.ref = ref
        self.idx = 0
        self.last_deps = None
        self.check_consistency()

    def check_consistency(self):
        for key in self.graph.vertices():
            try:
                self.ref[key]
            except KeyError as err:
                print(err, 'there is no ref')
        for key in self.ref:
            if key[-1] == 'f' and key[-2] != 'd':
                assert sum(self.ref[key][1]) == 4  # all feet must be fix
            if key[-1] == 'f' and key[-2] == 'd':
                assert sum(self.ref[key][1]) == 2  # 2 feet must be fix
            if key[-1] != 'f':
                try:
                    assert sum(self.ref[key][1]) == 2  # 2 feet must be fix
                except AssertionError:
                    print(key, 'sum is not equal 2')

    def get_next_reference(self, act_position, act_eps, xref, act_pose=None,
                           save_as_tikz=False, gait=False, save_png=False):
        xref = np.r_[xref]
        act_pos = np.r_[act_position]
        dpos = xref - act_pos
        act_dir = np.r_[np.cos(np.radians(act_eps)),
                        np.sin(np.radians(act_eps))]
        act_deps = calc_angle(dpos, act_dir)
        act_dist = np.linalg.norm(dpos)

        if act_dist < .5:
            pose_id = '000'
        else:
            if len(self.graph.get_children(self.pose)) > 1:
                verts = [v for v, weight in self.graph.get_children(self.pose)]
                plot = False if all(v[-1] == 'f' for v in verts) else True

                if plot:
                    figname = 'dec_'+str(self.idx)
                    plt.figure(figname)
                    draw_point_dir(act_pos, act_dir, size=20)
                    draw_point_dir(act_pos, [0, 0], size=20,
                                   label='ROBOT (%s)' % self.pose)
                    draw_point_dir(xref, [0, 0], size=20,
                                   label='GOAL')
                    act_pose.plot('gray')
                    if save_as_tikz:
                        plt.figure('tikz_'+figname)
                        draw_point_dir(act_pos, act_dir*.5, size=8)
                        draw_point_dir(act_pos, [.1, 0], size=1,
                                       label='ROBOT (%s)' % self.pose)
                        draw_point_dir(xref, [0, 0], size=8)
                    

                def suitability(translation_, rotation, v=None, plot=True):
                    translation_ = np.r_[translation_]
                    # translation is configured for eps=90
                    translation = rotate(translation_, np.radians(act_eps-90))
                    dir_ = np.r_[np.cos(np.radians(act_eps+rotation)),
                                 np.sin(np.radians(act_eps+rotation))]
                    pos_ = act_pos + translation
                    dist_ = np.linalg.norm(xref-pos_)
                    deps_ = calc_angle(xref-pos_, dir_)

                    # Label the thing
                    if plot:
                        plt.figure(figname)
                        draw_point_dir(pos_, dir_, label=v)
                        draw_line(act_pos+translation, xref)
                        if save_as_tikz:
                            plt.figure('tikz_'+figname)
                            draw_point_dir(pos_, dir_, label=v, size=5,
                                           colp='orange')
                            draw_line(act_pos+translation, xref)

                    return (dist_, deps_)

                deps = CandidateHandler()
                ddist = CandidateHandler()
                for child in self.graph.get_children(self.pose):
                    v, (translation, rotation) = child
                    dist_, deps_ = suitability(translation, rotation, v, plot)
                    deps[v] = round(deps_, 2)
                    ddist[v] = round(dist_, 2)

                if abs(act_deps) > 70:  # ganz falsche Richtung
                    _, pose_id = deps.minimum()
                else:
                    max_deps, _ = deps.maximum()
                    min_ddist, _ = ddist.maximum()

                    w = .5
                    dec = CandidateHandler()
                    for key in deps:
                        for dist, eps in zip(ddist[key], deps[key]):
                            dec[key] = w*dist/min_ddist + (1-w)*abs(eps)/max_deps
                    min_dec, pose_id = dec.minimum()

                if plot:
                    plt.figure(figname)
                    draw_point_dir(act_pos-act_dir*2, [0, 0], size=1,
                                   label='choose (%s)' % pose_id)
                    if save_as_tikz:
                        plt.figure('tikz_'+figname)
                        draw_point_dir(act_pos-act_dir*2, [0, 0], size=1,
                                       label='choose (%s)' % pose_id)
                        if gait:
                            gait.plot_markers(1, figname='tikz_'+figname)
                        plt.axis('off')
                        geckostr = act_pose.get_tikz_repr('gray')
                        save.save_plt_as_tikz('Out/pathplanner/'+figname+'.tex',
                                              geckostr)
                        plt.close('tikz_'+figname)
                    elif save_png:
                        plt.savefig('Out/pathplanner/'+figname+'.png',
                                    transparent=True, dpi=300)

                    plt.show()
            else:  # only 1 child
                pose_id, _ = self.graph.get_children(self.pose)[0]

        self.pose = pose_id
        self.idx += 1

        alpha, feet, process_time = self.__get_ref(pose_id)
        if pose_id[:3] == '111':
            if pose_id == '111':
                self.last_deps = act_deps
            if abs(self.last_deps) < 90:
                alpha[2] = self.last_deps
                print('something magic\ndeps:\t', act_deps, '\n', alpha)
        if pose_id[:3] == '114':
            if pose_id == '114':
                self.last_deps = act_deps
            if abs(self.last_deps) < 90:
                alpha[2] = self.last_deps
                print('something magic\ndeps:\t', act_deps, '\n', alpha)

        return alpha, feet, process_time, pose_id

    def __get_ref(self, pose_id):
        return self.ref[pose_id]


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


def normalize(vec):
    x, y = vec
    leng = np.sqrt(x**2 + y**2)
    return np.r_[x/leng, y/leng]


def calc_angle(vec1, vec2, rotate_angle=0., jump=0):
    theta = np.radians(rotate_angle)
    vec1 = rotate(vec1, theta)
    x1, y1 = vec1  # normalize(vec1)
    x2, y2 = vec2  # normalize(vec2)
    phi1 = np.arctan2(y1, x1)
    vec2 = rotate([x2, y2], -phi1+jump)
    phi2 = np.degrees(np.arctan2(vec2[1], vec2[0]) - jump)
    alpha = -phi2
    return alpha


class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given, an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour, cost in self.__graph_dict[vertex]:
                if (vertex, neighbour, cost) not in edges:
                    edges.append((vertex, neighbour, cost))
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

    def find_isolated_vertices(self):
        """ returns a list of isolated vertices. """
        graph = self.__graph_dict
        isolated = []
        for vertex in graph:
            print(isolated, vertex)
            if not graph[vertex]:
                isolated += [vertex]
        return isolated

    def find_path(self, start_vertex, end_vertex, path=[]):
        """ find a path from start_vertex to end_vertex
            in graph """
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex, _ in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to
            end_vertex in graph """
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex, _ in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex,
                                                     path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def get_children(self, vertex):
        return self.__graph_dict[vertex]
