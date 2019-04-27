""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""

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

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        vertex1 = edge.pop()
        if edge:
            # not a loop
            vertex2 = edge.pop()
        else:
            # a loop
            vertex2 = vertex1
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

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
        for vertex in graph[start_vertex]:
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
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, 
                                                     end_vertex, 
                                                     path)
                for p in extended_paths: 
                    paths.append(p)
        return paths

    def is_connected(self, 
                     vertices_encountered = None, 
                     start_vertex=None):
        """ determines if the graph is connected """
        if vertices_encountered is None:
            vertices_encountered = set()
        gdict = self.__graph_dict        
        vertices = gdict.keys() 
        if not start_vertex:
            # chosse a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_encountered.add(start_vertex)
        if len(vertices_encountered) != len(vertices):
            for vertex in gdict[start_vertex]:
                if vertex not in vertices_encountered:
                    if self.is_connected(vertices_encountered, vertex):
                        return True
        else:
            return True
        return False

 
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
        dot.edge(v, w, label=str(c))
    
    print(dot.source)
    dot.render('file_name', view=True)
    
    
if __name__ == "__main__":
    from graphviz import Digraph
    
    g = { '000' : [('100', ((0,0), 0)), ('010', ((0,0), 0))],
          '100' : [('000', ((0,0), 0)), ('110', ((0,0), 0)),
                   ('010', ((0,0), 0)), ('104', ((0,0), 0)),
                   ('101', ((0,0), 0)), ('102', ((0,0), 0))],
          '110' : [('111', ((0,0), 0))],
          '111' : [('110b', ((0,0), 0))],
          '110b' : [('100', ((0,0), 0))],
          '104' : [('105', ((0,0), 0))],
          '105' : [('106', ((0,0), 0)), ('107', ((0,0), 0))],
          '106' : [('105', ((0,0), 0))],
          '107' : [('100', ((0,0), 0))],
          '101' : [('100', ((0,0), 0))],
          '102' : [('103', ((0,0), 0))],
          '103' : [('102', ((0,0), 0)), ('100', ((0,0), 0))],

          '010' : [('000', ((0,0), 0)), ('110c', ((0,0), 0)),
                   ('100', ((0,0), 0)), ('014', ((0,0), 0)),
                   ('011', ((0,0), 0)), ('012', ((0,0), 0))],
          '110c' : [('112', ((0,0), 0))],
          '112' : [('110d', ((0,0), 0))],
          '110d' : [('010', ((0,0), 0))],
          '014' : [('015', ((0,0), 0))],
          '015' : [('016', ((0,0), 0)), ('017', ((0,0), 0))],
          '016' : [('015', ((0,0), 0))],
          '017' : [('010', ((0,0), 0))],
          '011' : [('010', ((0,0), 0))],
          '012' : [('013', ((0,0), 0))],
          '013' : [('012', ((0,0), 0)), ('010', ((0,0), 0))]
        }

    graph = Graph(g)
    print(graph)

    render_graph(graph)
