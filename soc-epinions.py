# soc-Epinions

def download_and_load_soc_graph():
    url = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
    file_name = "soc-Epinions1.txt.gz"

    if not os.path.exists(file_name):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_name)

    print("Processing and loading graph...")
    
    edges = []
    with gzip.open(file_name, 'rt') as f:
        for line in f:
            # Skip comments or lines that do not represent edges
            if line.startswith('#') or line.strip() == "":
                continue
            # Split and validate edge definition
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure it's a valid edge
                edges.append(tuple(map(int, parts)))  # Convert node IDs to integers
    
    # Create the graph from the processed edge list
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


# ### 1. Using differential privacy


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
graph = download_and_load_soc_graph()
preprocessed_graph = preprocess_graph(graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_graph)

# Print results
print("\nAnonymization Results:")
for key, value in results.items():
    print(f"{key}: {value}")


# ### 3. Synthetic Graph

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


print("\nApplying hybrid anonymization with synthetic graph...")
anonymized_graph, results = anonymize_and_evaluate_with_synthetic_graph(preprocessed_graph, epsilon=0.3)

print("\nFinal Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value}")
