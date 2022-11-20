
from typing import Any, Dict, List, Set, Tuple
from pprint import pprint

WEB_GRAPH_1 = [("A", "F"), ("B", "A"), ("B", "E"), ("C", "B"),("D","D"),("E","E"),("F","E"),("F","B")]

def get_all_nodes(web_graph: List[Tuple[Any, Any]]) -> Set[Any]:

    nodes = set()
    for (from_node, to_node) in web_graph:
        nodes.add(from_node)
        nodes.add(to_node)
    
    return nodes

def get_outlinks_num(web_graph: List[Tuple[Any, Any]]) -> Dict[Any, int]:

    outlinks = {node: 0 for node in get_all_nodes(web_graph)}
    for (from_node, to_node) in web_graph:
        outlinks[from_node] += 1    
    return outlinks


def pagerank(web_graph: List[Tuple[Any, Any]], q: float = 0.2, iterations: int = 3) -> Dict[Any, float]:

    nodes = get_all_nodes(web_graph)
    outlinks_num = get_outlinks_num(web_graph)
    inlinks = {node: [] for node in nodes}
    for (from_node, to_node) in web_graph:
        inlinks[to_node].append(from_node)
    
    for node, lnum in outlinks_num.items():
        if lnum == 0:
            for to_node in nodes:
                inlinks[to_node].append(node)
            outlinks_num[node] = len(nodes)
    
    pr = {node: 1/len(nodes) for node in nodes}
    
    for i in range(iterations):
        pr_old = pr.copy()
        for node in pr.keys():
            pr[node] = q / len(nodes)
            for from_node in inlinks[node]:
                pr[node] += (1 - q) * pr_old[from_node] / outlinks_num[from_node]

    return pr


pprint(pagerank(WEB_GRAPH_1,q=0.2,iterations=1))
