# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:25:25 2019

@author: ls
"""
import numpy as np


g = { '000' : [('100', ((.16,.25), -17.1)), ('010', ((-.16,.25), 17.1)),
               ('000', ((0,0), 0))],

      '100' : [('000', ((0,0), 0)), ('110', ((0,-.2), -90)),
               ('010', ((0,.71), 1)), ('104', ((0,.2), -60)),
               ('101', ((.1,.53), -14)), ('102', ((0,.4), -30))],
      '110' : [('111', None)],
      '111' : [('110b', None)],
      '110b' : [('100', None)],
      '104' : [('105', None)],
      '105' : [('106', ((0,.2), -60)), ('107', ((0,.4), 0))],
      '106' : [('105', None)],
      '107' : [('100', None)],
      '101' : [('100', None)],
      '102' : [('103', None)],
      '103' : [('102', ((0,.4), -30)), ('100', ((0,.6), 0))],

      '010' : [('000', ((0,0), 0)), ('110c', ((.0,-.25), 90)),
               ('100', ((0,.71), -1)), ('014', ((0,.2), 60)),
               ('011', ((-.1,.53), .14)), ('012', ((0,.4), 30))],
      '110c' : [('112', None)],
      '112' : [('110d', None)],
      '110d' : [('010', None)],
      '014' : [('015', None)],
      '015' : [('016', ((0,.2), 60)), ('017', ((0,.4), 0))],
      '016' : [('015', None)],
      '017' : [('010', None)],
      '011' : [('010', None)],
      '012' : [('013', None)],
      '013' : [('012', ((0,.4), 30)), ('010', ((0,0.6), 0))]
    }

ref = { 
      '000' : [[0, 0, -0, 0, 0], [1, 0, 0, 0]],
      '100' : [[0, 90, 90, 0, 90], [0, 1, 1, 0]],
      '110' : [[45, 45, 0, 45, 45], [1, 0, 0, 1]],
      '111' : [[45, 45, -90, 45, 45], [1, 1, 0, 0]],
      '110b' : [[90, 0, -90, 90, 0], [1, 0, 0, 1]],
      '104' : [[50, 30, 90, 30, 150], [1, 0, 0, 1]],
      '105' : [[124, 164, 152, 62, 221], [0, 1, 1, 0]],
      '106' : [[0, 0, 24, 0, 0], [1, 0, 0, 1]],
      '107' : [[30, 90, 80, 10, 10], [1, 0, 0, 1]],
      '101' : [[40, 1, -10, 60, 10], [1, 0, 0, 1]],
      '102' : [[48, 104, 114, 27, 124], [0, 1, 1, 0]],
      '103' : [[1, 72, 70, 1, 55], [1, 0, 0, 1]],

      '010' : [[90, 0, -90, 90, 0], [1, 0, 0, 1]],
      '110c' : [[45, 45, 0, 45, 45], [0, 1, 1, 0]],
      '112' : [[45, 45, 90, 45, 45], [1, 1, 0, 0]],
      '110d' : [[45, 45, 0, 45, 45], [0, 0, 1, 1]],
      '014' : [[30, 50, -90, 150, 30], [0, 1, 1, 0]],
      '015' : [[164, 124, -152, 221, 62], [1, 0, 0, 1]],
      '016' : [[0, 0, -24, 0, 0], [0, 1, 1, 0]],
      '017' : [[90, 30, -80, 10, 10], [0, 1, 1, 0]],
      '011' : [[1, 40, 10, 10, 60], [0, 1, 1, 0]],
      '012' : [[104, 48, -114, 124, 27], [1, 0, 0, 1]],
      '013' : [[72, 1, -70, 55, 1], [0, 1, 1, 0]]
      }


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return (c*vec[0]-s*vec[1], s*vec[0]+c*vec[1])


def normalize(vec):
    x, y = vec
    l = np.sqrt(x**2 + y**2)
    return x/l, y/l


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

    def get_next_reference(self, act_pose, xref):
        mx, my = act_pose.markers
        act_pos = (mx[1], my[1])
        act_eps = act_pose.x[-1]
        dpos = [xref[0]-act_pos[0], xref[1]-act_pos[1]]
        act_dir = (np.cos(np.radians(act_eps)), np.sin(np.radians(act_eps)))
        deps = calc_angle(dpos, act_dir)
        act_dist = np.sqrt(dpos[0]**2 + dpos[1]**2)

        candidate = {}
        for child in self.graph.get_children(self.pose):
            try:
                v, (dist, eps) = child
                dist = dist[1]
                suitability = abs(deps-eps)
                if v == '000':
                    if act_dist < 1:
                        suitability = 0
                    else:
                        suitability = 10000
            except TypeError:
                if child[1] is None:
                    suitability = 0
            candidate[v] = suitability
        pose_id = min(candidate, key=candidate.get)
        self.pose = pose_id
        print(pose_id)

        return self.get_alpha(pose_id), pose_id

    def get_alpha(self, pose_id):
        return self.ref[pose_id]
        



class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, an empty dictionary will be used
        """
        if graph_dict == None:
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
                extended_path = self.find_path(vertex, 
                                               end_vertex, 
                                               path)
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
                extended_paths = self.find_all_paths(vertex, 
                                                     end_vertex, 
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
    
    # print(dot.source)
    dot.render('file_name', view=True)
    
    
if __name__ == "__main__":
    from graphviz import Digraph
    
    g = { '000' : [('100', ((.16,.25), -17.1)), ('010', ((-.16,.25), 17.1)),
                   ('000', ((0,0), 0))],

          '100' : [('000', ((0,0), 0)), ('110', ((0,-.2), -90)),
                   ('010', ((0,.71), 1)), ('104', ((0,.2), -60)),
                   ('101', ((.1,.53), -14)), ('102', ((0,.4), -30))],
          '110' : [('111', None)],
          '111' : [('110b', None)],
          '110b' : [('100', None)],
          '104' : [('105', None)],
          '105' : [('106', ((0,.2), -60)), ('107', ((0,.4), 0))],
          '106' : [('105', None)],
          '107' : [('100', None)],
          '101' : [('100', None)],
          '102' : [('103', None)],
          '103' : [('102', ((0,.4), -30)), ('100', ((0,.6), 0))],

          '010' : [('000', ((0,0), 0)), ('110c', ((.0,-.25), 90)),
                   ('100', ((0,.71), -1)), ('014', ((0,.2), 60)),
                   ('011', ((-.1,.53), .14)), ('012', ((0,.4), 30))],
          '110c' : [('112', None)],
          '112' : [('110d', None)],
          '110d' : [('010', None)],
          '014' : [('015', None)],
          '015' : [('016', ((0,.2), 60)), ('017', ((0,.4), 0))],
          '016' : [('015', None)],
          '017' : [('010', None)],
          '011' : [('010', None)],
          '012' : [('013', None)],
          '013' : [('012', ((0,.4), 30)), ('010', ((0,0.6), 0))]
        }

    graph = Graph(g)
    print(graph.get_children('000'))

    render_graph(graph)
