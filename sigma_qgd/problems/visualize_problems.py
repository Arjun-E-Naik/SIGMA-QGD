import sys
from pathlib import Path

# Support running directly as a script as well as an imported module
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from .ansatz import ansatz_function
except (ImportError, ValueError):
    from sigma_qgd.problems.ansatz import ansatz_function
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pennylane as qml

def save_ansatz_image(qubits:int =4, reps:int = 1,filename:str = "ansatz_circuit.png"):

    dev = qml.device("default.qubit",wires=qubits)

    @qml.qnode(dev)
    def circuit(theta):
        ansatz_function(theta=theta, n=qubits, reps=reps)
        return qml.state()


    theta_vals = np.random.random(qubits * reps)

    fig, ax = qml.draw_mpl(circuit)(theta_vals)
    plt.suptitle(f"Ring Ansatz (Qubits: {qubits}, Reps: {reps})", fontsize=14)

    plt.savefig(filename,bbox_inches="tight",dpi = 200)

    plt.close()
    print(f"saved ansatz circuit diagram to:{filename}")


def save_hamilltonian_graph(qubits:int = 4,filename:str = "hamiltonian_graph.png"):

    G = nx.Graph()

    for i in range(qubits):
        G.add_node(i,lable=f"Qubit {i}")

    for i in range(qubits):
        j = (i+1) % qubits
        G.add_edge(i,j)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(5,5))

    nx.draw_networkx_nodes(G,pos,node_size=700,node_color='skyblue',edgecolors= 'black')
    nx.draw_networkx_edges(G,pos,width=2.5,edge_color='gray')
    nx.draw_networkx_labels(G,pos,font_size=10,font_weight='bold')
    plt.title("Hamiltonian interaction Graph (Ring Topology)",fontsize = 16)

    plt.axis('off')

    plt.savefig(filename,bbox_inches="tight",dpi=300)
    plt.close()
    print(f"saved Hamiltonian graph to: {filename}")


if __name__ == "__main__":
    n_qubits = 10

    save_ansatz_image(qubits=n_qubits,reps=2,filename="ansatz_graph.png")
    save_hamilltonian_graph(qubits=n_qubits,filename="hamiltonian_graph.png")
    print(f"successfully saved both graphs for {n_qubits} qubits.")