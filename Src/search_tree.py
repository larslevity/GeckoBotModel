# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:25:25 2019

@author: ls
"""
import numpy as np
import matplotlib.pyplot as plt


g = {  # manually tuned
     '000': [('100', ((.16, .25), -17.1)), ('010', ((-.16, .25), 17.1)),
             ('000', ((0, 0), 0))],

     '100': [('000', ((0, 0), 0)),
             ('010', ((0, .71), 1)), ('104', ((.2, .2), -90)),
             ('101', ((.1, .53), -30)), ('102', ((.15, .4), -60))],
     '110': [('111', None)],
     '111': [('110b', None)],
     '110b': [('100', None)],
     '104': [('105', None)],
     '105': [('106', ((0, .2), -130)), ('107', ((0, .4), -70))],
     '106': [('105', None)],
     '107': [('100', None)],
     '101': [('100', None)],
     '102': [('103', None)],
     '103': [('102', ((0, .4), -30)), ('100', ((0, .6), -15))],

     '010': [('000', ((0, 0), 0)),
             ('100', ((0, .71), -1)), ('014', ((-.2, .2), 90)),
             ('011', ((-.1, .53), 30)), ('012', ((-.15, .4), 60))],
     '110c': [('112', None)],
     '112': [('110d', None)],
     '110d': [('010', None)],
     '014': [('015', None)],
     '015': [('016', ((0, .2), 130)), ('017', ((0, .4), 70))],
     '016': [('015', None)],
     '017': [('010', None)],
     '011': [('010', None)],
     '012': [('013', None)],
     '013': [('012', ((0, .4), 30)), ('010', ((0, 0.6), 15))]
    }

g_ = {
     '000': [('100', ((.16, .25), -17.1)), ('010', ((-.16, .25), 17.1)),
             ('000', ((0, 0), 0))],

     '100': [('000', ((0, 0), 0)), ('110', ((.1, -.2), -90)),
             ('010', ((0, .71), 1)), ('104', ((.2, .2), -60)),
             ('101', ((.1, .53), -14)), ('102', ((.15, .4), -30))],
     '110': [('111', None)],
     '111': [('110b', None)],
     '110b': [('100', None)],
     '104': [('105', None)],
     '105': [('106', ((0, .2), -60)), ('107', ((0, .4), 0))],
     '106': [('105', None)],
     '107': [('100', None)],
     '101': [('100', None)],
     '102': [('103', None)],
     '103': [('102', ((0, .4), -30)), ('100', ((0, .6), 0))],

     '010': [('000', ((0, 0), 0)), ('110c', ((-.1, -.25), 90)),
             ('100', ((0, .71), -1)), ('014', ((-.2, .2), 60)),
             ('011', ((-.1, .53), 14)), ('012', ((-.15, .4), 30))],
     '110c': [('112', None)],
     '112': [('110d', None)],
     '110d': [('010', None)],
     '014': [('015', None)],
     '015': [('016', ((0, .2), 60)), ('017', ((0, .4), 0))],
     '016': [('015', None)],
     '017': [('010', None)],
     '011': [('010', None)],
     '012': [('013', None)],
     '013': [('012', ((0, .4), 30)), ('010', ((0, 0.6), 0))]
    }

ref = {
     '000': [[0, 0, -0, 0, 0], [1, 0, 0, 1]],
     '100': [[0, 90, 90, 0, 90], [0, 1, 1, 0]],
     '110': [[45, 45, 0, 45, 45], [1, 0, 0, 1]],
     '111': [[45, 45, -90, 45, 45], [1, 1, 0, 0]],
     '110b': [[90, 0, -90, 90, 0], [1, 0, 0, 1]],
     '104': [[50, 30, 90, 30, 150], [1, 0, 0, 1]],
     '105': [[124, 164, 152, 62, 221], [0, 1, 1, 0]],
     '106': [[0, 0, 24, 0, 0], [1, 0, 0, 1]],
     '107': [[30, 90, 80, 10, 10], [1, 0, 0, 1]],
     '101': [[40, 1, -10, 60, 10], [1, 0, 0, 1]],
     '102': [[48, 104, 114, 27, 124], [0, 1, 1, 0]],
     '103': [[1, 72, 70, 1, 55], [1, 0, 0, 1]],

     '010': [[90, 0, -90, 90, 0], [1, 0, 0, 1]],
     '110c': [[45, 45, 0, 45, 45], [0, 1, 1, 0]],
     '112': [[45, 45, 90, 45, 45], [1, 1, 0, 0]],
     '110d': [[45, 45, 0, 45, 45], [0, 0, 1, 1]],
     '014': [[30, 50, -90, 150, 30], [0, 1, 1, 0]],
     '015': [[164, 124, -152, 221, 62], [1, 0, 0, 1]],
     '016': [[0, 0, -24, 0, 0], [0, 1, 1, 0]],
     '017': [[90, 30, -80, 10, 10], [0, 1, 1, 0]],
     '011': [[1, 40, 10, 10, 60], [0, 1, 1, 0]],
     '012': [[104, 48, -114, 124, 27], [1, 0, 0, 1]],
     '013': [[72, 1, -70, 55, 1], [0, 1, 1, 0]]
      }


def draw_point_dir(point, direction, size=12, label=None):
    x1, y1 = point
    dx, dy = direction
    plt.plot([x1], [y1], marker='o', color='red', markersize=size)
    plt.plot([x1, x1+dx], [y1, y1+dy], color='blue')
    if label:
        plt.text(x1+.05, y1, label)
    plt.axis('equal')


def draw_line(point1, point2, color='gray', linestyle='dashed'):
    x1, y1 = point1
    x2, y2 = point2
    plt.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle)


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


class ReferenceGenerator(object):
    def __init__(self, pose='000', graph=g, ref=ref):
        self.graph = Graph(graph)
        self.pose = pose
        self.ref = ref
        self.idx = 0
        self.last_deps = None

    def get_next_reference(self, act_pose, xref):
        xref = np.r_[xref]
        mx, my = act_pose.markers
        act_pos = np.r_[mx[1], my[1]]
        act_eps = act_pose.x[-1]
        dpos = xref - act_pos
        act_dir = np.r_[np.cos(np.radians(act_eps)),
                        np.sin(np.radians(act_eps))]
        deps = calc_angle(dpos, act_dir)
        act_dist = np.linalg.norm(dpos)

        print('act eps:\t%s' % act_eps)
        print('act dir:\t%s' % act_dir)
        print('act dist:\t%s' % act_dist)
        print('delta eps:\t%s' % deps)

        plt.figure('dec'+str(self.idx))
        draw_point_dir(act_pos, act_dir, size=20,
                       label='HOME (%s)' % self.pose)
        draw_point_dir(xref, [0, 0], size=20,
                       label='GOAL')

        def suitability(translation_, rotation, v=None):
            translation_ = np.r_[translation_]
            # translation is configured for eps=90
            translation = rotate(translation_, np.radians(act_eps-90))
            dir_ = np.r_[np.cos(np.radians(act_eps+rotation)),
                         np.sin(np.radians(act_eps+rotation))]
            pos_ = act_pos + translation
            dist_ = np.linalg.norm(xref-pos_)
            deps_ = calc_angle(xref-pos_, dir_)

            # Label the thing
            draw_point_dir(pos_, dir_, label=v)
            draw_line(act_pos+translation, xref)

            return (dist_, deps_)

        if act_dist < 1:
            pose_id = '000'
        else:
            if len(self.graph.get_children(self.pose)) > 1:
                deps = {}
                ddist = {}
                for child in self.graph.get_children(self.pose):
                    v, (translation, rotation) = child
                    dist_, deps_ = suitability(translation, rotation, v)
                    deps[v] = deps_
                    ddist[v] = dist_

                min_deps = min([abs(deps[key]) for key in deps])
                max_deps = max([abs(deps[key]) for key in deps])
                min_ddist = min([ddist[key] for key in ddist])
                w = .5
                dec = {key: (w*ddist[key]/min_ddist
                             + (1-w)*abs(deps[key])/max_deps) for key in deps}
                print dec
                if min_deps > 70:  # ganz falsche Richtung
                    deps_ = {key: abs(deps[key]) for key in deps}
                    pose_id = min(deps_, key=deps_.get)
                else:
                    pose_id = min(dec, key=dec.get)
                    # pose_id = min(ddist, key=ddist.get)
                    # if abs(deps[pose_id]) > abs(self.last_deps):

                draw_point_dir(xref+np.r_[0, -1], [0, 0], size=1,
                               label='choose (%s)' % pose_id)

                self.last_deps = deps[pose_id]
            else:  # only 1 child
                pose_id, _ = self.graph.get_children(self.pose)[0]
    #        plt.show()

        self.pose = pose_id
        self.idx += 1
        print(pose_id)

        return self.get_alpha(pose_id), pose_id

    def get_alpha(self, pose_id):
        return self.ref[pose_id]


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


def render_graph(graph):
    """ requirements:
    pip install graphviz
    apt-get install graphviz
    """
    dot = Digraph()
    for v in graph.vertices():
        dot.node(v, v)

    for e in graph.edges():
        v = e[0]
        w = e[1]
        c = e[2]
        dot.edge(v, w, label=str(c) if c else None)

    dot.render('file_name', view=True)


if __name__ == "__main__":
    try:
        from graphviz import Digraph
        graph = Graph(g)
        print(graph.get_children('000'))
        render_graph(graph)
    except ImportError:
        print('Missing package gaphiviz')
        print('run: "pip install graphviz" and "apt-get install graphviz" ')

    import plot_fun as pf
    import kinematic_model as model

    alpha = [90, 0, -90, 90, 0]
    eps = 180
    F1 = (0, 0)
    initial_pose = model.set_initial_pose(alpha, eps, F1)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    ref = ReferenceGenerator('010')
    xref = (10, -2.5)

    def calc_dist(pose, xref):
        mx, my = pose.markers
        act_pos = np.r_[mx[1], my[1]]
        dpos = xref - act_pos
        return np.linalg.norm(dpos)

    i = 0
    while calc_dist(gait.poses[-1], xref) > 1:
        act_pose = gait.poses[-1]
        ref_new, pose_id = ref.get_next_reference(act_pose, xref)
        gait.append_pose(model.predict_next_pose(ref_new, act_pose))
        i += 1
        if i > 50:
            break

    gait.plot_gait()
    draw_point_dir(xref, [0, 0], size=20, label='GOAL1')
