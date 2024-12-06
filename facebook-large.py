# # facebook-large

def download_and_load_facebook_large_graph():
    url = "https://snap.stanford.edu/data/facebook_large.zip"
    zip_file_name = "facebook_large.zip"
    extracted_folder = "facebook_large"

    # Download the dataset if it doesn't exist
    if not os.path.exists(zip_file_name):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_file_name)

    # Extract the dataset if not already extracted
    if not os.path.exists(extracted_folder):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

    # Paths to the extracted files
    edges_file = os.path.join(extracted_folder, "musae_facebook_edges.csv")
    features_file = os.path.join(extracted_folder, "musae_facebook_features.json")
    target_file = os.path.join(extracted_folder, "musae_facebook_target.csv")

    print("Processing and loading graph...")
    
    # Load edges
    edges_df = pd.read_csv(edges_file)
    edges = list(zip(edges_df["id_1"], edges_df["id_2"]))

    # Create an undirected graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # Load node features (optional, for additional analysis)
    node_features = pd.read_json(features_file, orient="index")

    # Load node labels (optional, for classification tasks)
    node_labels = pd.read_csv(target_file)
    node_labels = dict(zip(node_labels["id"], node_labels["page_type"]))

    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Node features loaded: {node_features.shape[1]} features per node.")
    print(f"Node labels loaded: {len(node_labels)} nodes with labels.")
    
    return graph, node_features, node_labels

# Example usage
facebook_graph, facebook_node_features, facebook_node_labels = download_and_load_facebook_large_graph()


# In[ ]:


def preprocess_graph(graph):
    """
    Preprocess the facebook_large graph by removing isolated nodes, self-loops, and normalizing node IDs.
    """
    print("Starting graph preprocessing...")

    # Remove isolated nodes
    isolates = list(nx.isolates(graph))
    if isolates:
        print(f"Found {len(isolates)} isolated nodes. Removing them...")
        graph.remove_nodes_from(isolates)

    # Remove self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        print(f"Found {len(self_loops)} self-loops. Removing them...")
        graph.remove_edges_from(self_loops)

    # Ensure the graph is undirected
    if nx.is_directed(graph):
        print("Converting directed graph to undirected...")
        graph = nx.Graph(graph)

    # Normalize node IDs
    print("Normalizing node IDs...")
    mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)

    print(f"Preprocessing complete. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


def compute_graph_metrics(graph):
    """
    Compute basic and advanced metrics for the graph.
    """
    print("Computing graph metrics...")

    metrics = {
        "Number of Nodes": graph.number_of_nodes(),
        "Number of Edges": graph.number_of_edges(),
        "Density": nx.density(graph),
        "Transitivity": nx.transitivity(graph),  # Equivalent to global clustering coefficient
        "Average Clustering Coefficient": nx.average_clustering(graph),
    }

    # Connected Components
    connected_components = list(nx.connected_components(graph))
    metrics["Number of Connected Components"] = len(connected_components)
    metrics["Size of Largest Connected Component"] = max(len(c) for c in connected_components)

    # Centrality Measures
    print("Calculating centrality measures...")
    metrics["Max Degree Centrality"] = max(nx.degree_centrality(graph).values())
    metrics["Max Closeness Centrality"] = max(nx.closeness_centrality(graph).values())
    metrics["Max Betweenness Centrality"] = max(nx.betweenness_centrality(graph).values())

    # Diameter and Path Length (if connected)
    if nx.is_connected(graph):
        metrics["Diameter"] = nx.diameter(graph)
        metrics["Average Path Length"] = nx.average_shortest_path_length(graph)
    else:
        print("Graph is not connected. Skipping diameter and average path length calculations.")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


# In[13]:


import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


def measure_privacy_preservation(original_graph, anonymized_graph):
    """
    Measure the privacy preservation by calculating the edge re-identification rate.
    """
    print("Measuring privacy preservation...")

    original_edges = set(original_graph.edges())
    anonymized_edges = set(anonymized_graph.edges())

    # Calculate the percentage of edges re-identified
    common_edges = original_edges.intersection(anonymized_edges)
    reidentification_rate = len(common_edges) / len(original_edges) * 100

    print(f"Edge re-identification rate: {reidentification_rate:.2f}%")
    return reidentification_rate


def evaluate_data_utility(original_graph, anonymized_graph, original_features, anonymized_features, original_labels):
    """
    Evaluate the utility of the anonymized graph by comparing features and a simple classification task.
    """
    print("Evaluating data utility...")

    # Prepare data for classification
    degrees_original = dict(original_graph.degree())
    degrees_anonymized = dict(anonymized_graph.degree())

    # Use degree as a simple feature for classification
    nodes_original = list(degrees_original.keys())
    labels_original = [original_labels[node] for node in nodes_original]

    # Train-test split for the original graph
    X_train, X_test, y_train, y_test = train_test_split(
        list(degrees_original.values()), labels_original, test_size=0.3, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    clf.fit(np.array(X_train).reshape(-1, 1), y_train)
    y_pred = clf.predict(np.array(X_test).reshape(-1, 1))
    original_accuracy = accuracy_score(y_test, y_pred)

    # Evaluate the model on the anonymized graph
    anonymized_nodes = list(degrees_anonymized.keys())
    anonymized_labels = [original_labels.get(node, 0) for node in anonymized_nodes]
    anonymized_predictions = clf.predict(
        np.array(list(degrees_anonymized.values())).reshape(-1, 1)
    )
    anonymized_accuracy = accuracy_score(anonymized_labels, anonymized_predictions)

    print(f"Original Graph Classification Accuracy: {original_accuracy:.2f}")
    print(f"Anonymized Graph Classification Accuracy: {anonymized_accuracy:.2f}")

    return original_accuracy, anonymized_accuracy


def evaluate_robustness(original_graph, anonymized_graph):
    """
    Evaluate the robustness of the anonymized graph against structural attacks.
    """
    print("Evaluating robustness...")

    original_degrees = sorted([degree for _, degree in original_graph.degree()])
    anonymized_degrees = sorted([degree for _, degree in anonymized_graph.degree()])

    # Measure similarity of degree distributions
    degree_difference = sum(abs(o - a) for o, a in zip(original_degrees, anonymized_degrees))
    robustness_score = 1 - (degree_difference / sum(original_degrees))

    print(f"Robustness (degree distribution similarity): {robustness_score:.2f}")
    return robustness_score


def evaluate_graph_similarity(original_graph, anonymized_graph):
    """
    Evaluate similarity metrics between the original and anonymized graphs.
    """
    print("Evaluating graph similarity...")
    results = {}

    # Degree Distribution Correlation
    original_degrees = np.array([degree for _, degree in original_graph.degree()])
    anonymized_degrees = np.array([degree for _, degree in anonymized_graph.degree()])
    degree_correlation = np.corrcoef(original_degrees, anonymized_degrees)[0, 1]

    # Clustering Coefficient Correlation
    original_clustering = np.array(list(nx.clustering(original_graph).values()))
    anonymized_clustering = np.array(list(nx.clustering(anonymized_graph).values()))
    clustering_correlation = np.corrcoef(original_clustering, anonymized_clustering)[0, 1]

    results["Degree Distribution Correlation"] = degree_correlation
    results["Clustering Coefficient Correlation"] = clustering_correlation

    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    return results


def evaluate_graph_anonymization(
    original_graph, anonymized_graph, original_features, anonymized_features, original_labels
):
    """
    Evaluate various aspects of graph anonymization, including privacy preservation,
    data utility, robustness, and structural similarity.
    """
    print("Evaluating graph anonymization...")

    print("Measuring privacy preservation...")
    reidentification_rate = measure_privacy_preservation(original_graph, anonymized_graph)

    print("Evaluating data utility...")
    original_accuracy, anonymized_accuracy = evaluate_data_utility(
        original_graph, anonymized_graph, original_features, anonymized_features, original_labels
    )

    print("Evaluating robustness...")
    robustness_score = evaluate_robustness(original_graph, anonymized_graph)

    print("Evaluating graph similarity...")
    similarity_metrics = evaluate_graph_similarity(original_graph, anonymized_graph)

    results = {
        "Re-Identification Rate": reidentification_rate,
        "Original Accuracy": original_accuracy,
        "Anonymized Accuracy": anonymized_accuracy,
        "Robustness Score": robustness_score,
        **similarity_metrics,
    }

    return results


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


def anonymize_and_evaluate(graph, features, labels):
    epsilon = 0.3  # Moderate privacy budget for better utility-privacy balance

    # Apply differential privacy to the graph
    print("Anonymizing graph with differential privacy...")
    perturbed_graph = add_differential_privacy_to_degrees(graph, epsilon)

    # Add topology-preserving edges
    print("Adding topology-preserving edges...")
    perturbed_graph = add_topology_preserving_edges(perturbed_graph, k=2)

    # Evaluate the anonymized graph
    print("Evaluating anonymized graph...")
    results = evaluate_graph_anonymization(graph, perturbed_graph, features, features, labels)
    return results


# Load and preprocess the Facebook Large graph
facebook_graph, facebook_features, facebook_labels = download_and_load_facebook_large_graph()
preprocessed_facebook_graph = preprocess_graph(facebook_graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_facebook_graph, facebook_features, facebook_labels)

# Print results
print("\nAnonymization Results:")
for key, value in results.items():
    print(f"{key}: {value}")


# ### 2. Using Hybrid Anonymization

# In[ ]:


def add_differential_privacy_to_degrees(graph, epsilon=0.3):
    """
    Add differential privacy to the node degrees of a graph with topology-aware noise.
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
        noise = np.random.laplace(0, adjusted_scale, 1)[0]
        noisy_degree = max(0, int(true_degree + noise))

        # Add or remove edges to achieve noisy degree
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
    """
    features = {}
    features['Degree'] = dict(graph.degree())
    features['Clustering Coefficient'] = nx.clustering(graph)
    features['Betweenness Centrality'] = nx.betweenness_centrality(graph)
    features['Closeness Centrality'] = nx.closeness_centrality(graph)
    features['Triangles'] = nx.triangles(graph)
    return features


def anonymize_and_evaluate(graph, features, labels):
    """
    Perform hybrid anonymization and evaluate the anonymized graph.
    """
    epsilon = 0.3  # Privacy budget

    # Apply hybrid anonymization
    print("Applying hybrid anonymization...")
    anonymized_graph = hybrid_anonymization(graph, epsilon)

    # Extract robust features for evaluation
    print("Extracting robust features for utility evaluation...")
    original_features = extract_robust_features(graph)
    anonymized_features = extract_robust_features(anonymized_graph)

    # Evaluate the anonymized graph
    print("Evaluating anonymized graph...")
    results = evaluate_graph_anonymization(graph, anonymized_graph, features, features, labels)

    # Include feature similarity in results
    feature_similarity = {
        feature: np.corrcoef(
            list(original_features[feature].values()),
            list(anonymized_features[feature].values())
        )[0, 1] for feature in original_features
    }
    results['Feature Similarity'] = feature_similarity

    return results


# Load the Facebook Large graph
facebook_graph, facebook_features, facebook_labels = download_and_load_facebook_large_graph()

# Preprocess the graph
preprocessed_facebook_graph = preprocess_graph(facebook_graph)

# Perform anonymization and evaluation
results = anonymize_and_evaluate(preprocessed_facebook_graph, facebook_features, facebook_labels)

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
    print("Generating synthetic graph from degree distribution...")
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
    print("Applying hybrid anonymization...")
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


def anonymize_and_evaluate_with_synthetic_graph(original_graph, features, labels, epsilon=0.3):
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
    results = evaluate_graph_anonymization(original_graph, anonymized_graph, features, features, labels)
    results["Anonymization Time"] = anonymization_time

    return anonymized_graph, results


# Load Facebook Large dataset
facebook_graph, facebook_features, facebook_labels = download_and_load_facebook_large_graph()

# Preprocess the Facebook Large graph
preprocessed_facebook_graph = preprocess_graph(facebook_graph)

# Apply hybrid anonymization with synthetic graph
print("\nApplying hybrid anonymization with synthetic graph...")
anonymized_facebook_graph, facebook_results = anonymize_and_evaluate_with_synthetic_graph(
    preprocessed_facebook_graph, facebook_features, facebook_labels, epsilon=0.3
)

# Print Final Evaluation Results
print("\nFinal Evaluation Results:")
for key, value in facebook_results.items():
    if key == 'Feature Similarity':
        print(f"{key}:")
        for feature, similarity in value.items():
            print(f"  {feature}: {similarity:.4f}")
    else:
        print(f"{key}: {value}")


# ### 4. Synthetic Graph with DCSBM

# In[ ]:


# Helper Functions
def generate_synthetic_graph_dcsbm(original_graph, intra_prob=0.3, inter_prob=0.01):
    """
    Generate a synthetic graph using Degree-Corrected Stochastic Block Model (DCSBM).
    """
    print("Detecting communities...")
    communities = list(greedy_modularity_communities(original_graph))
    print(f"Detected {len(communities)} communities.")

    synthetic_graph = nx.Graph()
    synthetic_graph.add_nodes_from(original_graph.nodes())

    print("Adding intra-community edges...")
    for community in communities:
        community_nodes = list(community)
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                if np.random.random() < intra_prob:
                    synthetic_graph.add_edge(community_nodes[i], community_nodes[j])

    print("Adding inter-community edges...")
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
    """
    refined_graph = anonymized_graph.copy()

    original_clustering = nx.clustering(original_graph)
    anonymized_clustering = nx.clustering(refined_graph)

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


def anonymize_and_evaluate_with_refinements(original_graph, features, labels, epsilon_values, intra_prob=0.3, inter_prob=0.01):
    results = {}
    for epsilon in epsilon_values:
        print(f"\nTesting with epsilon = {epsilon}...")

        # Generate synthetic graph
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
        original_accuracy, anonymized_accuracy = evaluate_data_utility(
            original_graph, refined_graph, features, features, labels
        )

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


# Load Facebook Large dataset
facebook_graph, facebook_features, facebook_labels = download_and_load_facebook_large_graph()

# Preprocess the Facebook Large graph
preprocessed_facebook_graph = preprocess_graph(facebook_graph)

# Anonymization and evaluation with refinements
epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]
intra_prob = 0.4
inter_prob = 0.02

print("\nAnonymizing and evaluating Facebook Large graph...")
results = anonymize_and_evaluate_with_refinements(
    preprocessed_facebook_graph, facebook_features, facebook_labels, epsilon_values, intra_prob=intra_prob, inter_prob=inter_prob
)

# Print Final Evaluation Results
print("\nAnonymization Results:")
for epsilon, metrics in results.items():
    print(f"Epsilon: {epsilon}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")