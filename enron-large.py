# enron-large

def download_and_load_email_enron_graph():
    url = "https://snap.stanford.edu/data/email-Enron.txt.gz"
    file_name = "email-Enron.txt.gz"

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
graph = download_and_load_email_enron_graph()
preprocessed_graph = preprocess_graph(graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_graph)

# Print results
print("\nAnonymization Results:")
for key, value in results.items():
    print(f"{key}: {value}")


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


print("\nApplying hybrid anonymization with synthetic graph...")
anonymized_graph, results = anonymize_and_evaluate_with_synthetic_graph(preprocessed_graph, epsilon=0.3)

print("\nFinal Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value}")


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
    original_graph = download_and_load_email_enron_graph()
    preprocessed_graph = preprocess_graph(original_graph)
    return preprocessed_graph

def main():
    graph = download_and_preprocess_graph()

    epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Test different epsilon values
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