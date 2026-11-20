import networkx as nx
from data_preparation.make_network import construct_graph


def getGraphMetrics(sangerID):
    """ 
    Prints several connectivity metrics of a gene graph based on the input
    SangerID, which is a cell line
    """
    
    graph = construct_graph(sanger_id=sangerID)
    
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(graph)}")
    print(f"Density: {nx.density(graph)}")

    if nx.is_connected(graph):
        print(f"Diameter: {nx.diameter(graph)}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(graph)}")
        print(f"Average node connectivity: {nx.average_node_connectivity(graph)}")
    else:
        subgraphs = nx.connected_components(graph)
        largest_cc = graph.subgraph(max(subgraphs, key=len)).copy()
        print(f"Number of connected components: {nx.number_connected_components(graph)}")
        print(f"Largest component size: {largest_cc.number_of_nodes()}")
        print(f"Largest component fraction: {largest_cc.number_of_nodes() / graph.number_of_nodes()}")
        print(f"Largest component diameter: {nx.diameter(largest_cc)}")
        print(f"Largest component avg path length: {nx.average_shortest_path_length(largest_cc)}")

    print(f"Average clustering: {nx.average_clustering(graph)}")
    print(f"Global clustering : {nx.transitivity(graph)}")

getGraphMetrics("SIDM00046")

    



    
    