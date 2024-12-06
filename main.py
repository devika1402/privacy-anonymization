"""
##########################################################
# Graph Anonymization and Evaluation Framework
# Author: Devika Rajasekar
# Date: 06 Dec 2024

# Description:
# This script provides a comprehensive framework for anonymizing 
# and analyzing graph datasets using various techniques, including:
# 1. Random Perturbation
# 2. Differential Privacy
# 3. Hybrid Anonymization
# 4. Synthetic Graph Generation
# 5. Degree-Corrected Stochastic Block Models (DCSBM)

# Key Features:
# - Implements anonymization techniques that preserve graph utility while enhancing privacy.
# - Measures privacy risks and utility loss through robust evaluation metrics.
# - Visualizes original and anonymized graphs for intuitive comparisons.
# - Includes pre-defined methods for processing, anonymizing, and evaluating graph datasets.

# Main Libraries Used:
# - NetworkX: For graph processing and metrics.
# - NumPy: For numerical computations and noise addition.
# - Scikit-learn: For evaluating data utility through classification tasks.
# - Matplotlib: For visualization of graphs and metrics.
# - Node2Vec: For graph embeddings and representation learning.

# How to Use:
# 1. Define or download a graph dataset using the appropriate functions.
# 2. Preprocess the graph to remove anomalies like isolated nodes or self-loops.
# 3. Apply anonymization techniques as required.
# 4. Evaluate anonymized graphs using metrics like re-identification rate, accuracy, robustness, and modularity.
# 5. Visualize results using provided plotting functions.

# Notes:
# - Ensure the required datasets and libraries are installed before running the script.
# - Customize privacy parameters (e.g., epsilon) based on the desired balance of utility and privacy.
##########################################################
"""

import os
import time
import random
import gzip
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.community import modularity, greedy_modularity_communities

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from node2vec import Node2Vec
from sklearn.manifold import TSNE

from scipy.stats import powerlaw
from matplotlib.patches import Patch



def download_and_load_graph():
    url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
    file_name = "ca-GrQc.txt.gz"

    if not os.path.exists(file_name):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_name)

    print("Processing and loading graph...")
    edges = []
    with gzip.open(file_name, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # Skip comments or empty lines
            parts = line.strip().split()
            try:
                edges.append(tuple(map(int, parts)))  # Convert node IDs to integers only if valid
            except ValueError:
                continue  # Skip lines where conversion to int fails

    graph = nx.Graph()
    graph.add_edges_from(edges)
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph



def preprocess_graph(graph):
    print("Starting graph preprocessing...")

    # Handle Missing or Incomplete Data
    print("Checking for isolated nodes...")
    isolates = list(nx.isolates(graph))
    if isolates:
        print(f"Found {len(isolates)} isolated nodes. Removing them...")
        graph.remove_nodes_from(isolates)

    print("Checking for self-loops...")
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        print(f"Found {len(self_loops)} self-loops. Removing them...")
        graph.remove_edges_from(self_loops)

    print("Checking for duplicate edges...")
    # Since NetworkX automatically handles multiple edges between two nodes in a simple Graph,
    # this step is conceptual unless using a MultiGraph
    if isinstance(graph, nx.MultiGraph):
        original_edge_count = sum(1 for _ in graph.edges())
        graph = nx.Graph(graph)  # Converts MultiGraph to Graph, thus removing duplicate edges
        new_edge_count = graph.number_of_edges()
        duplicate_edges_count = original_edge_count - new_edge_count
        print(f"Found {duplicate_edges_count} duplicate edges. Converted to simple graph.")
    else:
        print("No duplicate edges found; the graph is already a simple graph.")

    # Standardize Node and Edge IDs
    print("Normalizing node IDs...")
    mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)

    # Ensure the graph is undirected
    print("Ensuring graph is undirected...")
    if nx.is_directed(graph):
        graph = graph.to_undirected()
        print("Converted to undirected graph.")
    else:
        print("Graph is already undirected.")

    print(f"Preprocessing complete. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph




def compute_graph_metrics(graph):
    print("Computing Graph Metrics...")

    # Basic Graph Metrics
    properties = {
        "Number of Nodes": graph.number_of_nodes(),
        "Number of Edges": graph.number_of_edges(),
        "Density": nx.density(graph),
    }

    # Connected Components
    connected_components = list(nx.connected_components(graph))
    properties["Number of Connected Components"] = len(connected_components)
    largest_component_size = max(len(component) for component in connected_components) if connected_components else 0
    properties["Size of Largest Connected Component"] = largest_component_size

    # Advanced Metrics
    print("Calculating centrality measures...")
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    properties["Max Degree Centrality"] = max(degree_centrality.values())
    properties["Max Closeness Centrality"] = max(closeness_centrality.values())
    properties["Max Betweenness Centrality"] = max(betweenness_centrality.values())

    # Clustering Coefficient
    properties["Clustering Coefficient"] = nx.average_clustering(graph)

    # Check if graph is connected and calculate Diameter and Path Length
    if nx.is_connected(graph):
        properties["Diameter"] = nx.diameter(graph)
        properties["Average Path Length"] = nx.average_shortest_path_length(graph)
        properties["Triangles"] = sum(nx.triangles(graph).values()) // 3

    # Print all metrics
    print("\n".join([f"{k}: {v}" for k, v in properties.items()]))

    return properties


# In[ ]:


# Measure re-identification risk
def measure_privacy_preservation(original_graph, perturbed_graph):
    print("Measuring privacy preservation...")
    
    original_edges = set(original_graph.edges())
    perturbed_edges = set(perturbed_graph.edges())

    # Calculate the percentage of edges re-identified
    common_edges = original_edges.intersection(perturbed_edges)
    reidentification_rate = len(common_edges) / len(original_edges) * 100

    print(f"Edge re-identification rate: {reidentification_rate:.2f}%")
    return reidentification_rate

# Evaluate structural and attribute preservation
def evaluate_data_utility(original_graph, perturbed_graph):
    print("Evaluating data utility...")

    def extract_features(graph):
        degrees = dict(graph.degree())
        return [(node, degrees[node]) for node in graph.nodes()]

    original_features = extract_features(original_graph)
    perturbed_features = extract_features(perturbed_graph)

    nodes_original, degrees_original = zip(*original_features)
    nodes_perturbed, degrees_perturbed = zip(*perturbed_features)

    # Generate synthetic labels for classification (based on degree threshold)
    threshold = sum(degrees_original) / len(degrees_original)
    labels_original = [1 if degree > threshold else 0 for degree in degrees_original]

    # Train-test split and model training on the original graph
    X_train, X_test, y_train, y_test = train_test_split(degrees_original, labels_original, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit([[x] for x in X_train], y_train)
    y_pred = clf.predict([[x] for x in X_test])
    original_accuracy = accuracy_score(y_test, y_pred)

    # Evaluate the trained model on the perturbed graph
    y_perturbed_pred = clf.predict([[x] for x in degrees_perturbed])
    anonymized_accuracy = accuracy_score(labels_original, y_perturbed_pred)

    print(f"Original accuracy: {original_accuracy:.2f}")
    print(f"Anonymized accuracy: {anonymized_accuracy:.2f}")
    return original_accuracy, anonymized_accuracy

# Anonymization Cost: Measure computational cost
def measure_anonymization_cost(anonymization_function, graph):
    print("Measuring anonymization cost...")
    start_time = time.time()
    perturbed_graph = anonymization_function(graph)
    elapsed_time = time.time() - start_time
    print(f"Time taken for anonymization: {elapsed_time:.2f} seconds")
    return elapsed_time, perturbed_graph

# Robustness: Test against structural attacks
def evaluate_robustness(original_graph, perturbed_graph):
    print("Evaluating robustness...")

    original_degrees = sorted([degree for _, degree in original_graph.degree()])
    perturbed_degrees = sorted([degree for _, degree in perturbed_graph.degree()])

    # Measure similarity of degree distributions
    degree_difference = sum(abs(o - p) for o, p in zip(original_degrees, perturbed_degrees))
    robustness_score = 1 - (degree_difference / sum(original_degrees))

    print(f"Robustness (degree distribution similarity): {robustness_score:.2f}")
    return robustness_score


# In[ ]:


def evaluate_graph_similarity(original_graph, anonymized_graph):
    """
    Evaluate additional similarity metrics between the original and anonymized graphs.
    """
    results = {}

    # Degree Distribution Correlation
    original_degrees = [degree for _, degree in original_graph.degree()]
    anonymized_degrees = [degree for _, degree in anonymized_graph.degree()]
    results["Degree Distribution Correlation"] = (
        np.corrcoef(original_degrees, anonymized_degrees)[0, 1]
        if len(original_degrees) > 1
        else None
    )

    # Clustering Coefficient Correlation
    original_clustering = list(nx.clustering(original_graph).values())
    anonymized_clustering = list(nx.clustering(anonymized_graph).values())
    results["Clustering Coefficient Correlation"] = (
        np.corrcoef(original_clustering, anonymized_clustering)[0, 1]
        if len(original_clustering) > 1
        else None
    )

    # Triangles Similarity
    original_triangles = list(nx.triangles(original_graph).values())
    anonymized_triangles = list(nx.triangles(anonymized_graph).values())
    results["Triangles Similarity"] = (
        np.corrcoef(original_triangles, anonymized_triangles)[0, 1]
        if len(original_triangles) > 1
        else None
    )

    # Modularity Difference
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity

        original_communities = list(greedy_modularity_communities(original_graph))
        anonymized_communities = list(greedy_modularity_communities(anonymized_graph))
        original_modularity = modularity(original_graph, original_communities)
        anonymized_modularity = modularity(anonymized_graph, anonymized_communities)
        results["Modularity Difference"] = abs(original_modularity - anonymized_modularity)
    except ImportError:
        results["Modularity Difference"] = None

    return results


# In[ ]:


def evaluate_graph_anonymization(original_graph, anonymization_function, *args, **kwargs):
    print("Applying anonymization...")
    start_time = time.time()
    anonymized_graph = anonymization_function(original_graph, *args, **kwargs)
    anonymization_time = time.time() - start_time
    print(f"Anonymization completed in {anonymization_time:.2f} seconds.")

    print("Evaluating privacy preservation...")
    reidentification_rate = measure_privacy_preservation(original_graph, anonymized_graph)

    print("Evaluating data utility...")
    original_accuracy, anon_accuracy = evaluate_data_utility(original_graph, anonymized_graph)

    print("Evaluating robustness...")
    robustness_score = evaluate_robustness(original_graph, anonymized_graph)

    print("Evaluating structural similarity...")
    similarity_metrics = evaluate_graph_similarity(original_graph, anonymized_graph)

    results = {
        "Anonymization Time": anonymization_time,
        "Re-Identification Rate": reidentification_rate,
        "Original Accuracy": original_accuracy,
        "Anonymized Accuracy": anon_accuracy,
        "Robustness Score": robustness_score,
        **similarity_metrics,  # Adding all the similarity metrics
    }

    return results


# ### 0. Applying random pertubation

# In[ ]:


# Function to apply random perturbation
def apply_random_perturbation(graph, perturbation_rate=0.1):
    """
    Apply random perturbation by randomly adding and removing edges.

    :param graph: A NetworkX graph.
    :param perturbation_rate: Fraction of edges to perturb.
    :return: A perturbed graph.
    """
    perturbed_graph = graph.copy()
    edges = list(perturbed_graph.edges())
    nodes = list(perturbed_graph.nodes())
    num_perturbations = int(perturbation_rate * len(edges))

    # Randomly remove edges
    for _ in range(num_perturbations):
        if edges:
            edge_to_remove = random.choice(edges)
            perturbed_graph.remove_edge(*edge_to_remove)
            edges.remove(edge_to_remove)

    # Randomly add edges
    for _ in range(num_perturbations):
        u, v = random.sample(nodes, 2)
        if not perturbed_graph.has_edge(u, v):
            perturbed_graph.add_edge(u, v)

    return perturbed_graph


# ### 1. Using differential privacy

# In[ ]:


def add_differential_privacy_to_degrees(graph, epsilon=0.3):
    """
    Add differential privacy to the node degrees of a graph with topology-aware noise.

    :param graph: A NetworkX graph.
    :param epsilon: Privacy budget, a smaller epsilon means more privacy.
    :return: A perturbed graph with noisy degrees and edges.
    """
    sensitivity = 1  # Sensitivity of the degree function is 1
    scale = sensitivity / epsilon

    # Create a copy of the graph to perturb
    perturbed_graph = nx.Graph()
    perturbed_graph.add_nodes_from(graph.nodes())

    # Topology-aware noise: Adjust noise based on centrality metrics
    degree_centrality = nx.degree_centrality(graph)
    for node in graph.nodes():
        true_degree = graph.degree[node]
        node_centrality = degree_centrality[node]

        # Scale noise inversely with centrality to obfuscate high-centrality nodes less
        adjusted_scale = scale / (1 + node_centrality)
        noise = np.random.laplace(0, adjusted_scale, 1)[0]  # Generate Laplace noise
        noisy_degree = max(0, int(true_degree + noise))  # Ensure non-negative degree

        # Add or remove edges to achieve noisy degree
        neighbors = list(graph.neighbors(node))
        random.shuffle(neighbors)
        # Add edges
        while len(neighbors) < noisy_degree:
            potential_neighbor = random.choice(list(graph.nodes()))
            if node != potential_neighbor and not perturbed_graph.has_edge(node, potential_neighbor):
                perturbed_graph.add_edge(node, potential_neighbor)
                neighbors.append(potential_neighbor)
        # Remove excess edges if noisy degree is less than the true degree
        while len(neighbors) > noisy_degree:
            neighbor = neighbors.pop()
            if perturbed_graph.has_edge(node, neighbor):
                perturbed_graph.remove_edge(node, neighbor)

    return perturbed_graph

def add_topology_preserving_edges(graph, k=2):
    """
    Add edges to preserve connectivity while obfuscating structure selectively.

    :param graph: A NetworkX graph.
    :param k: Number of random edges to add for each low-centrality node.
    :return: A graph with additional random edges.
    """
    degree_centrality = nx.degree_centrality(graph)
    for node in graph.nodes():
        if degree_centrality[node] < 0.1:  # Focus on low-centrality nodes
            for _ in range(k):
                potential_neighbor = random.choice(list(graph.nodes()))
                if node != potential_neighbor and not graph.has_edge(node, potential_neighbor):
                    graph.add_edge(node, potential_neighbor)

    return graph

def anonymize_and_evaluate(graph):
    epsilon = 0.3  # Moderate privacy budget for better utility-privacy balance

    # Apply differential privacy to the graph
    print("Anonymizing graph with differential privacy...")
    perturbed_graph = add_differential_privacy_to_degrees(graph, epsilon)

    # Add topology-preserving edges
    print("Adding topology-preserving edges...")
    perturbed_graph = add_topology_preserving_edges(perturbed_graph, k=2)

    # Evaluate the anonymized graph
    results = evaluate_graph_anonymization(graph, add_differential_privacy_to_degrees, epsilon=epsilon)
    return results

# Download and preprocess the graph
graph = download_and_load_graph()
preprocessed_graph = preprocess_graph(graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_graph)

# Print results
print("\nAnonymization Results:")
for key, value in results.items():
    print(f"{key}: {value}")


# ### Visualization function

# In[22]:


def visualize_graph_comparison(original_graph, random_perturbed_graph, differential_private_graph):
    """
    Visualize the original graph, randomly perturbed graph, and differentially private graph.

    :param original_graph: Original NetworkX graph.
    :param random_perturbed_graph: Randomly perturbed graph.
    :param differential_private_graph: Differentially private graph.
    """
    plt.figure(figsize=(18, 6))

    # Original graph
    plt.subplot(1, 3, 1)
    nx.draw(
        original_graph,
        with_labels=False,
        node_size=10,
        node_color="blue",
        edge_color="gray"
    )
    plt.title("Original Graph")

    # Randomly perturbed graph
    plt.subplot(1, 3, 2)
    nx.draw(
        random_perturbed_graph,
        with_labels=False,
        node_size=10,
        node_color="red",
        edge_color="gray"
    )
    plt.title("Randomly Perturbed Graph")

    # Differentially private graph
    plt.subplot(1, 3, 3)
    nx.draw(
        differential_private_graph,
        with_labels=False,
        node_size=10,
        node_color="green",
        edge_color="gray"
    )
    plt.title("Differentially Private Graph")

    plt.savefig("comparison1.png", dpi=300)
    plt.show()

# Apply random perturbation
random_perturbed_graph = apply_random_perturbation(preprocessed_graph, perturbation_rate=0.1)

# Apply differential privacy
differential_private_graph = add_differential_privacy_to_degrees(preprocessed_graph, epsilon=0.3)

# Visualize the comparison
visualize_graph_comparison(preprocessed_graph, random_perturbed_graph, differential_private_graph)


# ### 2. Using Hybrid Anonymization

# In[ ]:


def add_differential_privacy_to_degrees(graph, epsilon=0.3):
    """
    Add differential privacy to the node degrees of a graph with topology-aware noise.

    :param graph: A NetworkX graph.
    :param epsilon: Privacy budget, a smaller epsilon means more privacy.
    :return: A perturbed graph with noisy degrees and edges.
    """
    sensitivity = 1  # Sensitivity of the degree function is 1
    scale = sensitivity / epsilon

    # Create a copy of the graph to perturb
    perturbed_graph = nx.Graph()
    perturbed_graph.add_nodes_from(graph.nodes())

    # Topology-aware noise: Adjust noise based on centrality metrics
    degree_centrality = nx.degree_centrality(graph)
    for node in graph.nodes():
        true_degree = graph.degree[node]
        node_centrality = degree_centrality[node]

        # Scale noise inversely with centrality to obfuscate high-centrality nodes less
        adjusted_scale = scale / (1 + node_centrality)
        noise = np.random.laplace(0, adjusted_scale, 1)[0]  # Generate Laplace noise
        noisy_degree = max(0, int(true_degree + noise))  # Ensure non-negative degree

        # Add or remove edges to achieve noisy degree
        neighbors = list(graph.neighbors(node))
        random.shuffle(neighbors)
        # Add edges
        while len(neighbors) < noisy_degree:
            potential_neighbor = random.choice(list(graph.nodes()))
            if node != potential_neighbor and not perturbed_graph.has_edge(node, potential_neighbor):
                perturbed_graph.add_edge(node, potential_neighbor)
                neighbors.append(potential_neighbor)
        # Remove excess edges if noisy degree is less than the true degree
        while len(neighbors) > noisy_degree:
            neighbor = neighbors.pop()
            if perturbed_graph.has_edge(node, neighbor):
                perturbed_graph.remove_edge(node, neighbor)

    return perturbed_graph

def add_topology_preserving_edges(graph, k=2):
    """
    Add edges to preserve connectivity while obfuscating structure selectively.

    :param graph: A NetworkX graph.
    :param k: Number of random edges to add for each low-centrality node.
    :return: A graph with additional random edges.
    """
    degree_centrality = nx.degree_centrality(graph)
    for node in graph.nodes():
        if degree_centrality[node] < 0.1:  # Focus on low-centrality nodes
            for _ in range(k):
                potential_neighbor = random.choice(list(graph.nodes()))
                if node != potential_neighbor and not graph.has_edge(node, potential_neighbor):
                    graph.add_edge(node, potential_neighbor)

    return graph

def hybrid_anonymization(graph, epsilon=0.3):
    """
    Combine differential privacy with edge-swapping for hybrid anonymization.

    :param graph: A NetworkX graph.
    :param epsilon: Privacy budget for differential privacy.
    :return: An anonymized graph.
    """
    perturbed_graph = add_differential_privacy_to_degrees(graph, epsilon)

    # Edge-swapping to preserve degree distributions while obfuscating structure
    edges = list(perturbed_graph.edges())
    random.shuffle(edges)
    num_swaps = len(edges) // 10  # Swap 10% of edges

    for _ in range(num_swaps):
        edge1 = random.choice(edges)
        edge2 = random.choice(edges)
        u1, v1 = edge1
        u2, v2 = edge2

        # Ensure no self-loops or duplicate edges are created
        if len(set([u1, v1, u2, v2])) == 4:
            if perturbed_graph.has_edge(u1, v1) and perturbed_graph.has_edge(u2, v2):
                perturbed_graph.remove_edge(u1, v1)
                perturbed_graph.remove_edge(u2, v2)
                perturbed_graph.add_edge(u1, u2)
                perturbed_graph.add_edge(v1, v2)

    return perturbed_graph

def extract_robust_features(graph):
    """
    Extract robust features from the graph for utility evaluation.

    :param graph: A NetworkX graph.
    :return: A dictionary of features.
    """
    features = {}
    features['Degree'] = dict(graph.degree())
    features['Clustering Coefficient'] = nx.clustering(graph)
    features['Betweenness Centrality'] = nx.betweenness_centrality(graph)
    features['Closeness Centrality'] = nx.closeness_centrality(graph)
    features['Triangles'] = nx.triangles(graph)
    return features

def anonymize_and_evaluate(graph):
    epsilon = 0.3  # Moderate privacy budget for better utility-privacy balance

    # Apply hybrid anonymization to the graph
    print("Applying hybrid anonymization...")
    anonymized_graph = hybrid_anonymization(graph, epsilon)

    # Extract robust features for evaluation
    print("Extracting robust features for utility evaluation...")
    original_features = extract_robust_features(graph)
    anonymized_features = extract_robust_features(anonymized_graph)

    # Evaluate the anonymized graph
    results = evaluate_graph_anonymization(graph, hybrid_anonymization, epsilon=epsilon)

    # Include feature similarity in results
    feature_similarity = {
        feature: np.corrcoef(list(original_features[feature].values()),
                             list(anonymized_features[feature].values()))[0, 1]
        for feature in original_features
    }
    results['Feature Similarity'] = feature_similarity

    return results

# Download and preprocess the graph
graph = download_and_load_graph()
preprocessed_graph = preprocess_graph(graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_graph)

# Print results
print("\nAnonymization Results:")
for key, value in results.items():
    if key == 'Feature Similarity':
        print(f"{key}:")
        for feature, similarity in value.items():
            print(f"  {feature}: {similarity:.4f}")
    else:
        print(f"{key}: {value}")


# In[23]:


def visualize_noise_scaling(graph, perturbed_graph, centrality):
    """
    Visualize the original graph with centrality and perturbed graph with noise scaling.
    """
    # Create positions for consistent layout
    pos = nx.spring_layout(graph, seed=42)

    # Node sizes proportional to centrality
    node_sizes = [centrality[node] * 1000 for node in graph.nodes()]

    # Original Graph Visualization
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color="lightblue",
        edge_color="gray",
        font_size=8,
        font_weight="bold",
    )
    plt.title("Original Graph with Degree Centrality")

    # Perturbed Graph Visualization
    plt.subplot(1, 2, 2)
    nx.draw(
        perturbed_graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color="lightgreen",
        edge_color="gray",
        font_size=8,
        font_weight="bold",
    )

    # Add noise arrows to nodes in the perturbed graph
    for node in graph.nodes():
        x, y = pos[node]
        noise_value = abs(graph.degree[node] - perturbed_graph.degree[node])
        if noise_value > 0:
            plt.arrow(
                x,
                y,
                random.uniform(-0.05, 0.05),
                random.uniform(-0.05, 0.05),
                color="red",
                head_width=0.05 * noise_value,
                alpha=0.6,
                length_includes_head=True,
            )

    plt.title("Perturbed Graph with Noise Scaling")

    plt.tight_layout()
    plt.show()


# In[25]:


# Compute degree centrality for visualization
degree_centrality = nx.degree_centrality(graph)

# Apply differential privacy
perturbed_graph = add_differential_privacy_to_degrees(graph)

# Visualize the noise scaling
visualize_noise_scaling(graph, perturbed_graph, degree_centrality)


# ### 3. Synthetic Graph

# In[ ]:


# Helper Functions
def generate_synthetic_graph_from_degree_distribution(original_graph):
    """
    Generate a synthetic graph with a matching degree distribution using the Configuration Model.
    """
    degree_sequence = [degree for _, degree in original_graph.degree()]
    synthetic_graph = nx.configuration_model(degree_sequence)
    synthetic_graph = nx.Graph(synthetic_graph)  # Convert to simple graph
    synthetic_graph.remove_edges_from(nx.selfloop_edges(synthetic_graph))
    return synthetic_graph

def add_differential_privacy_to_degrees(graph, epsilon=0.3):
    """
    Add differential privacy to the node degrees of a graph with topology-aware noise.
    """
    sensitivity = 1  # Sensitivity of the degree function is 1
    scale = sensitivity / epsilon

    perturbed_graph = nx.Graph()
    perturbed_graph.add_nodes_from(graph.nodes())
    degree_centrality = nx.degree_centrality(graph)

    for node in graph.nodes():
        true_degree = graph.degree[node]
        node_centrality = degree_centrality[node]
        adjusted_scale = scale / (1 + node_centrality)
        noise = np.random.laplace(0, adjusted_scale, 1)[0]
        noisy_degree = max(0, int(true_degree + noise))

        neighbors = list(graph.neighbors(node))
        random.shuffle(neighbors)

        while len(neighbors) < noisy_degree:
            potential_neighbor = random.choice(list(graph.nodes()))
            if node != potential_neighbor and not perturbed_graph.has_edge(node, potential_neighbor):
                perturbed_graph.add_edge(node, potential_neighbor)
                neighbors.append(potential_neighbor)

        while len(neighbors) > noisy_degree:
            neighbor = neighbors.pop()
            if perturbed_graph.has_edge(node, neighbor):
                perturbed_graph.remove_edge(node, neighbor)

    return perturbed_graph

def hybrid_anonymization(graph, epsilon=0.3):
    """
    Combine differential privacy with edge-swapping for hybrid anonymization.
    """
    perturbed_graph = add_differential_privacy_to_degrees(graph, epsilon)
    edges = list(perturbed_graph.edges())
    random.shuffle(edges)
    num_swaps = len(edges) // 10

    for _ in range(num_swaps):
        edge1 = random.choice(edges)
        edge2 = random.choice(edges)
        u1, v1 = edge1
        u2, v2 = edge2

        if len(set([u1, v1, u2, v2])) == 4:
            if perturbed_graph.has_edge(u1, v1) and perturbed_graph.has_edge(u2, v2):
                perturbed_graph.remove_edge(u1, v1)
                perturbed_graph.remove_edge(u2, v2)
                perturbed_graph.add_edge(u1, u2)
                perturbed_graph.add_edge(v1, v2)

    return perturbed_graph

def anonymize_and_evaluate_with_synthetic_graph(original_graph, epsilon=0.3):
    """
    Main workflow to anonymize using synthetic graph and evaluate the results.
    """
    print("Generating synthetic graph...")
    synthetic_graph = generate_synthetic_graph_from_degree_distribution(original_graph)

    print("Applying differential privacy to synthetic graph...")
    start_time = time.time()
    anonymized_graph = hybrid_anonymization(synthetic_graph, epsilon)
    anonymization_time = time.time() - start_time
    print(f"Anonymization completed in {anonymization_time:.2f} seconds.")

    print("\nEvaluating graph anonymization...")
    results = evaluate_graph_anonymization(original_graph, hybrid_anonymization, epsilon=epsilon)
    results["Anonymization Time"] = anonymization_time

    return anonymized_graph, results

# Main Execution
def main():
    print("Downloading and loading the graph...")
    graph = download_and_load_graph()

    print("Preprocessing the graph...")
    preprocessed_graph = preprocess_graph(graph)

    print("\nApplying hybrid anonymization with synthetic graph...")
    anonymized_graph, results = anonymize_and_evaluate_with_synthetic_graph(preprocessed_graph, epsilon=0.3)

    print("\nFinal Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()


# ### 4. Synthetic Graph with DCSBM

# In[ ]:


def generate_synthetic_graph_dcsbm(original_graph, intra_prob=0.3, inter_prob=0.01):
    """
    Generate a synthetic graph using Degree-Corrected Stochastic Block Model (DCSBM).

    :param original_graph: A NetworkX graph.
    :param intra_prob: Probability of edges within communities.
    :param inter_prob: Probability of edges between communities.
    :return: A synthetic graph preserving community structures and degree distribution.
    """
    # Detect communities using modularity-based method
    communities = list(greedy_modularity_communities(original_graph))
    community_map = {node: i for i, community in enumerate(communities) for node in community}

    # Group nodes by community
    community_sizes = [len(c) for c in communities]

    synthetic_graph = nx.Graph()
    synthetic_graph.add_nodes_from(original_graph.nodes())

    # Add intra-community edges
    for community in communities:
        community_nodes = list(community)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if np.random.random() < intra_prob:
                    synthetic_graph.add_edge(community_nodes[i], community_nodes[j])

    # Add inter-community edges
    for i, community_i in enumerate(communities):
        for j, community_j in enumerate(communities):
            if i >= j:
                continue
            for node_i in community_i:
                for node_j in community_j:
                    if np.random.random() < inter_prob:
                        synthetic_graph.add_edge(node_i, node_j)

    return synthetic_graph

def add_differential_privacy_to_degrees(graph, epsilon=0.3):
    """
    Add differential privacy to the node degrees of a graph with topology-aware noise.

    :param graph: A NetworkX graph.
    :param epsilon: Privacy budget, a smaller epsilon means more privacy.
    :return: A perturbed graph with noisy degrees and edges.
    """
    sensitivity = 1  # Sensitivity of the degree function is 1
    scale = sensitivity / epsilon

    perturbed_graph = nx.Graph()
    perturbed_graph.add_nodes_from(graph.nodes())

    degree_centrality = nx.degree_centrality(graph)
    for node in graph.nodes():
        true_degree = graph.degree[node]
        node_centrality = degree_centrality[node]

        adjusted_scale = scale / (1 + node_centrality)
        noise = np.random.laplace(0, adjusted_scale, 1)[0]
        noisy_degree = max(0, int(true_degree + noise))

        neighbors = list(graph.neighbors(node))
        random.shuffle(neighbors)
        while len(neighbors) < noisy_degree:
            potential_neighbor = random.choice(list(graph.nodes()))
            if node != potential_neighbor and not perturbed_graph.has_edge(node, potential_neighbor):
                perturbed_graph.add_edge(node, potential_neighbor)
                neighbors.append(potential_neighbor)
        while len(neighbors) > noisy_degree:
            neighbor = neighbors.pop()
            if perturbed_graph.has_edge(node, neighbor):
                perturbed_graph.remove_edge(node, neighbor)

    return perturbed_graph

def post_process_graph(anonymized_graph, original_graph):
    """
    Refine the anonymized graph to improve its similarity to the original graph.

    :param anonymized_graph: The anonymized NetworkX graph.
    :param original_graph: The original NetworkX graph.
    :return: A refined anonymized graph.
    """
    refined_graph = anonymized_graph.copy()

    original_clustering = nx.clustering(original_graph)
    anonymized_clustering = nx.clustering(refined_graph)

    # Adjust clustering coefficients
    for node in refined_graph.nodes():
        if anonymized_clustering[node] < original_clustering[node]:
            potential_neighbors = list(set(refined_graph.nodes()) - set(refined_graph.neighbors(node)) - {node})
            if potential_neighbors:
                neighbor_to_add = random.choice(potential_neighbors)
                refined_graph.add_edge(node, neighbor_to_add)
        elif anonymized_clustering[node] > original_clustering[node]:
            neighbors = list(refined_graph.neighbors(node))
            if neighbors:
                neighbor_to_remove = random.choice(neighbors)
                refined_graph.remove_edge(node, neighbor_to_remove)

    return refined_graph

def anonymize_and_evaluate_with_refinements(original_graph, epsilon_values, intra_prob=0.3, inter_prob=0.01):
    results = {}
    for epsilon in epsilon_values:
        print(f"Testing with epsilon = {epsilon}...")

        # Generate synthetic graph
        print("Generating synthetic graph using DCSBM...")
        synthetic_graph = generate_synthetic_graph_dcsbm(original_graph, intra_prob=intra_prob, inter_prob=inter_prob)

        # Apply differential privacy and measure time
        print("Applying differential privacy...")
        start_time = time.time()
        perturbed_graph = add_differential_privacy_to_degrees(synthetic_graph, epsilon)
        anonymization_time = time.time() - start_time
        print(f"Anonymization Time: {anonymization_time:.2f} seconds")

        # Post-process the graph
        print("Post-processing the anonymized graph...")
        refined_graph = post_process_graph(perturbed_graph, original_graph)

        # Evaluate metrics
        print("Evaluating similarity and additional metrics...")

        # Measure degree distribution correlation, clustering coefficient, and modularity
        degree_corr = np.corrcoef(
            [d for _, d in original_graph.degree()],
            [d for _, d in refined_graph.degree()]
        )[0, 1]
        clustering_corr = np.corrcoef(
            list(nx.clustering(original_graph).values()),
            list(nx.clustering(refined_graph).values())
        )[0, 1]
        modularity_diff = abs(
            modularity(original_graph, greedy_modularity_communities(original_graph)) -
            modularity(refined_graph, greedy_modularity_communities(refined_graph))
        )

        # Measure triangles similarity
        triangles_similarity = np.corrcoef(
            list(nx.triangles(original_graph).values()),
            list(nx.triangles(refined_graph).values())
        )[0, 1]

        # Re-identification rate
        original_edges = set(original_graph.edges())
        refined_edges = set(refined_graph.edges())
        common_edges = original_edges.intersection(refined_edges)
        re_identification_rate = len(common_edges) / len(original_edges) * 100

        # Original and anonymized accuracy
        original_accuracy, anonymized_accuracy = evaluate_data_utility(original_graph, refined_graph)

        # Robustness score
        robustness_score = evaluate_robustness(original_graph, refined_graph)

        # Consolidate results
        results[epsilon] = {
            "Anonymization Time": anonymization_time,
            "Re-Identification Rate": re_identification_rate,
            "Original Accuracy": original_accuracy,
            "Anonymized Accuracy": anonymized_accuracy,
            "Robustness Score": robustness_score,
            "Degree Distribution Correlation": degree_corr,
            "Clustering Coefficient Correlation": clustering_corr,
            "Triangles Similarity": triangles_similarity,
            "Modularity Difference": modularity_diff
        }

    return results

# Load and preprocess the graph
def download_and_preprocess_graph():
    original_graph = download_and_load_graph()
    preprocessed_graph = preprocess_graph(original_graph)
    return preprocessed_graph

def main():
    graph = download_and_preprocess_graph()

    epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Testing different epsilon values
    intra_prob = 0.4  # Adjusted to better preserve intra-community structure
    inter_prob = 0.02  # Adjusted to better preserve inter-community structure

    results = anonymize_and_evaluate_with_refinements(graph, epsilon_values, intra_prob=intra_prob, inter_prob=inter_prob)

    print("\nAnonymization Results:")
    for epsilon, metrics in results.items():
        print(f"Epsilon: {epsilon}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()


# In[ ]:


## Visualization

def create_dcsbm_schematic(original_graph, intra_prob=0.4, inter_prob=0.02, top_k=5):
    """
    Create an improved schematic representation of the Community-Based DCSBM Anonymization process.

    :param original_graph: A NetworkX graph.
    :param intra_prob: Probability of edges within communities.
    :param inter_prob: Probability of edges between communities.
    :param top_k: Number of largest communities to display separately.
    """
    # Community Detection
    communities = list(greedy_modularity_communities(original_graph))
    sorted_communities = sorted(communities, key=len, reverse=True)
    top_communities = sorted_communities[:top_k]
    other_nodes = set(node for community in sorted_communities[top_k:] for node in community)

    # Assign community IDs for top communities
    community_map = {}
    for idx, community in enumerate(top_communities):
        for node in community:
            community_map[node] = idx
    for node in other_nodes:
        community_map[node] = top_k  # Assign remaining nodes to "Other"

    # Colors for top communities and "Other"
    community_colors = [
        community_map[node] for node in original_graph.nodes()
    ]

    #  Synthetic Graph Generation
    synthetic_graph = nx.Graph()
    synthetic_graph.add_nodes_from(original_graph.nodes())

    # Add intra-community edges
    for community in top_communities:
        community_nodes = list(community)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if random.random() < intra_prob:
                    synthetic_graph.add_edge(community_nodes[i], community_nodes[j])

    # Add inter-community edges
    for i, community_i in enumerate(top_communities):
        for j, community_j in enumerate(top_communities):
            if i >= j:
                continue
            for node_i in community_i:
                for node_j in community_j:
                    if random.random() < inter_prob:
                        synthetic_graph.add_edge(node_i, node_j)

    # Plot the Schematic
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Adjust node sizes dynamically based on community size
    node_sizes = [
        100 if community_map[node] < top_k else 20 for node in original_graph.nodes()
    ]

    # Original Graph
    pos_original = nx.spring_layout(original_graph, seed=42)
    nx.draw(
        original_graph,
        pos_original,
        node_color=community_colors,
        cmap=plt.cm.tab10,
        node_size=node_sizes,
        ax=axes[0],
        edge_color="lightgray",
    )
    axes[0].set_title("Original Graph with Top Communities")

    # Community Detection
    pos_community = pos_original  # Use the same layout for consistency
    nx.draw(
        original_graph,
        pos_community,
        node_color=community_colors,
        cmap=plt.cm.tab10,
        node_size=node_sizes,
        ax=axes[1],
        edge_color="lightgray",
    )
    axes[1].set_title("Top Detected Communities")

    # Synthetic Graph
    pos_synthetic = nx.spring_layout(synthetic_graph, seed=42)
    nx.draw(
        synthetic_graph,
        pos_synthetic,
        node_color=community_colors,
        cmap=plt.cm.tab10,
        node_size=node_sizes,
        ax=axes[2],
        edge_color="lightgray",
        alpha=0.7,  # Make edges semi-transparent
    )
    axes[2].set_title("Synthetic Graph with Intra- and Inter-Community Edges")

    # Legend
    legend_labels = [f"Community {i}" for i in range(top_k)] + ["Other"]
    legend_patches = [
        Patch(color=plt.cm.tab10(i / (top_k + 1)), label=label)
        for i, label in enumerate(legend_labels)
    ]
    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(1.2, 1.05),
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig("dcsbm_schematic.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    graph = download_and_preprocess_graph()
    intra_prob = 0.4  # Adjusted to better preserve intra-community structure
    inter_prob = 0.02
    create_dcsbm_schematic(graph, intra_prob=intra_prob, inter_prob=inter_prob, top_k=5)