import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nun
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from collections import defaultdict
import dataset
import model, utils, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
'''from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase'''
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train(hyperparams=None, data_path='embedding/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    ##neo4j_uri = "neo4j+s://bb7d3bb8.databases.neo4j.io"
    neo4j_uri = "neo4j+s://293a7652.databases.neo4j.io"
    neo4j_user = "neo4j"
    ##neo4j_password = "0vZCoYqO6E9YkZRSFsdKPwHcziXu1-b0h8O9edAzWjM"
    neo4j_password = "RW3qSjBDIWEvjwgw2BN7-3xHQ6Qr9vbsrN7HjlN5MIM"
    
    reactome_file_path = "embedding/data/NCBI2Reactome.csv"
    output_file_path = "embedding/data/NCBI_pathway_map.csv"
    gene_names_file_path = "embedding/data/gene_names.csv"
    pathway_map = create_pathway_map(reactome_file_path, output_file_path)
    gene_id_to_name_mapping, gene_id_to_symbol_mapping = read_gene_names(gene_names_file_path)
    
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = dataset.PathwayDataset(data_path)
    ds_train = [ds[0]]
    ds_valid = [ds[1]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    ##criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = 'embedding/results/node_embeddings/'
    os.makedirs(results_path, exist_ok=True)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels_initial) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        stid_dic_initial= {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            stid_dic_initial[nx_graph.nodes[node]['stId']] = node_embeddings_initial[node_to_index_initial[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if cluster not in first_node_stId_in_cluster_initial:
                first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        print('first_node_stId_in_cluster_initial-------------------------------\n', first_node_stId_in_cluster_initial)
        stid_list = list(first_node_stId_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, stid_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)
    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)
      
    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                preds = (logits.sigmoid() > 0.5).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)

            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    preds = (logits.sigmoid() > 0.5).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1 = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels)

    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        loss_path = os.path.join(results_path, f'loss_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'f1_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(results_path, f'max_f1_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(results_path, f'matrix_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
 
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)

    save_path_pca = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap_= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix = os.path.join(results_path, f'matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_.png')
    
    cluster_stId_dict = {}  # Dictionary to store clusters and corresponding stIds
    significant_stIds = []  # List to store significant stIds
    clusters_with_significant_stId = {}  # Dictionary to store clusters and corresponding significant stIds
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            stid_dic[nx_graph.nodes[node]['stId']] = node_embeddings[node_to_index[node]]
            # Check if the node's significance is 'significant' and add its stId to the list
            
            ##print("node_embeddings[node_to_index[node]-------------------\n",graph.ndata['significance'][node_to_index[node]].item())
            if nx_graph.nodes[node]['significance'] == 'significant':
                significant_stIds.append(nx_graph.nodes[node]['stId'])
                ##print("significant_stIds-------------------\n",significant_stIds)
                ##print("nx_graph.nodes[node]['significance']-------------------\n",nx_graph.nodes[node]['significance'])
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = nx_graph.nodes[node]['stId']
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                
            # Populate cluster_stId_dict
            if cluster not in cluster_stId_dict:
                cluster_stId_dict[cluster] = []
            cluster_stId_dict[cluster].append({
                    'stId': nx_graph.nodes[node]['stId'],
                    'name': nx_graph.nodes[node]['name']
                })

            # Populate clusters_with_significant_stId
            if cluster not in clusters_with_significant_stId:
                clusters_with_significant_stId[cluster] = []
            
            ##print('significant_stIds-------------------\n',significant_stIds)
            if nx_graph.nodes[node]['stId'] in significant_stIds:
                ##clusters_with_significant_stId[cluster].append(nx_graph.nodes[node]['stId'])
                clusters_with_significant_stId[cluster].append({
                    'stId': nx_graph.nodes[node]['stId'],
                    'name': nx_graph.nodes[node]['name']
                })
                    
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'stId': nx_graph.nodes[node]['stId'],
                'significance': graph.ndata['significance'][node_to_index[node]].item(),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
        
        print(first_node_stId_in_cluster)
        stid_list = list(first_node_stId_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, stid_list, save_path_pca)
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    ##summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Best Val Score: {max_f1_valid}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Silhouette Score_ini: {silhouette_avg_}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"
    summary += f"Davies-Bouldin Index_ini: {davies_bouldin_}\n"

    save_file = os.path.join(results_path, f'dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)

    
    
    graph_train, graph_test = utils.create_embedding_with_markers()  

    # Get stid_mapping from save_graph_to_neo4j
    stid_mapping = utils.get_stid_mapping(graph_train)

    top_10_clusters = find_top_10_clusters_with_lowest_distance_and_connected_genes(all_embeddings, cluster_labels, cluster_stId_dict, pathway_map)

    output_file = os.path.join(results_path, f'significant_pathways_in_clusters_dim{out_feats}_lay{num_layers}_epo{num_epochs}.json')

    # Call the function with the top 10 clusters
    save_unique_genes_in_pathways(cluster_stId_dict, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, top_10_clusters, output_file)


    ##min_distance_cluster_id = find_cluster_with_lowest_distance_and_connected_genes(all_embeddings, cluster_labels, clusters_with_significant_stId, pathway_map)

    # Call the function with only the selected cluster
    ##save_unique_genes_in_pathways(clusters_with_significant_stId, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, min_distance_cluster_id)

    ##save_to_neo4j(graph_train, stid_dic, stid_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
 
    return model_path

def find_top_10_clusters_with_lowest_distance_and_connected_genes(embeddings, labels, cluster_stId_dict, pathway_map):
    cluster_distances = {}
    valid_clusters = set()

    for cluster_id in set(labels):
        cluster_indices = [i for i, x in enumerate(labels) if x == cluster_id]
        cluster_embeddings = embeddings[cluster_indices]

        if len(cluster_embeddings) > 1:
            distance_matrix = pairwise_distances(cluster_embeddings)
            within_cluster_distance = distance_matrix.sum() / 2
            cluster_distances[cluster_id] = within_cluster_distance

            # Check if the cluster has connected genes
            stIds_names = cluster_stId_dict.get(cluster_id, [])

            ##print('stIds_names=============\n', stIds_names)

            
            stIds = [entry['stId'] for entry in stIds_names]
            ##print('stIds-----------------------\n',stIds)
            names = [entry['name'] for entry in stIds_names]


            # Check for connected genes
            if isinstance(stIds, list):
                for single_stId in stIds:
                    genes = pathway_map.get(single_stId, [])
                    if genes:  # Check if there are connected genes
                        valid_clusters.add(cluster_id)
                        break
            else:
                genes = pathway_map.get(stIds, [])
                if genes:  # Check if there are connected genes
                    valid_clusters.add(cluster_id)

    # Filter out clusters without connected genes
    valid_cluster_distances = {cid: dist for cid, dist in cluster_distances.items() if cid in valid_clusters}

    if not valid_cluster_distances:
        raise ValueError("No valid clusters with connected genes found")

    # Get the top 10 clusters with the lowest distances
    top_10_clusters = sorted(valid_cluster_distances, key=valid_cluster_distances.get)[:10]
    return top_10_clusters

def save_unique_genes_in_pathways(cluster_stId_dict, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, top_clusters, output_file):
    # Gather significant pathways and their unique connected genes
    significant_pathways_in_clusters = {}

    for cluster_id in top_clusters:
        stIds_names = cluster_stId_dict.get(cluster_id, [])
        stId_name_pairs = [(entry['stId'], entry['name'], entry['weight'], entry['significance']) for entry in stIds_names]
 
        pathway_details = {}

        for stId, name, weight, significance in stId_name_pairs:
            if 'cancer' in name.lower():  # Only consider pathways with 'cancer' in their name
                ##print('name=========================\n',name)
                connected_genes = []
                genes = pathway_map.get(stId, [])
                unique_genes = set()  # To store unique genes for the pathway

                for gene_id in genes:
                    gene_name = gene_id_to_name_mapping.get(gene_id, None)
                    gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                    if gene_name and gene_symbol:  # Ensure that both name and symbol exist
                        connected_genes.append({
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "gene_symbol": gene_symbol
                        })
                        unique_genes.add(gene_symbol)

                if connected_genes:  # Only add pathways with connected genes
                    pathway_details[stId] = {
                        "name": name,
                        "weight": weight,
                        "significance": significance,
                        "connected_genes": connected_genes,
                        "unique_genes": list(unique_genes)
                    }

        if pathway_details:  # Only add clusters with pathway details
            significant_pathways_in_clusters[str(cluster_id)] = pathway_details

    # Save the significant pathways with connected unique genes to a JSON file
    with open(output_file, 'w') as f:
        json.dump(significant_pathways_in_clusters, f, indent=4)

def save_unique_genes_in_pathways_ori(clusters_with_significant_stId, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping):
    # Gather significant pathways and their unique connected genes
    significant_pathways_in_clusters = {}

    for cluster_id, stIds in clusters_with_significant_stId.items():
        pathway_details = {}

        if isinstance(stIds, list):
            # Handle the case where stIds is a list
            for single_stId in stIds:
                connected_genes = []
                genes = pathway_map.get(single_stId, [])
                unique_genes = set()  # To store unique genes for the pathway

                for gene_id in genes:
                    gene_name = gene_id_to_name_mapping.get(gene_id, None)
                    gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                    if gene_name and gene_symbol:  # Ensure that both name and symbol exist
                        connected_genes.append({
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "gene_symbol": gene_symbol
                        })
                        unique_genes.add(gene_symbol)
                
                if connected_genes:  # Only add pathways with connected genes
                    pathway_details[single_stId] = {
                        "connected_genes": connected_genes,
                        "unique_genes": list(unique_genes)
                    }
        else:
            # Handle the case where stIds is a single value
            connected_genes = []
            genes = pathway_map.get(stIds, [])
            unique_genes = set()  # To store unique genes for the pathway

            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name and gene_symbol:  # Ensure that both name and symbol exist
                    connected_genes.append({
                        "gene_id": gene_id,
                        "gene_name": gene_name,
                        "gene_symbol": gene_symbol
                    })
                    unique_genes.add(gene_symbol)
            
            if connected_genes:  # Only add pathways with connected genes
                pathway_details[stIds] = {
                    "connected_genes": connected_genes,
                    "unique_genes": list(unique_genes)
                }
        
        if pathway_details:  # Only add clusters with pathway details
            significant_pathways_in_clusters[str(cluster_id)] = pathway_details

    # Save the significant pathways with connected unique genes to a JSON file
    output_dir = 'embedding/results/node_embeddings'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'significant_pathways_in_clusters_with_unique_genes.json')

    with open(output_file, 'w') as f:
        json.dump(significant_pathways_in_clusters, f, indent=4)
        
##def save_to_neo4j(graph, stid_dic, stid_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password, clusters_with_significant_stId, results_path, cluster_labels, node_to_index, nx_graph, node_embeddings, first_node_stId_in_cluster):
def save_to_neo4j(graph, stid_dic, stid_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in stid_dic:
            embedding = stid_dic[node_id].tolist()  
            stId = stid_mapping[node_id]  # Access stId based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Pathway {embedding: $embedding, stId: $stId, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, stId=stId, name=name, weight=weight, significance=significance
            )

            # Create gene nodes and relationships
            ##genes = get_genes_by_pathway_stid(node_id, reactome_file, gene_names_file)
            genes = pathway_map.get(node_id, [])


            ##print('stid_to_gene_info=========================-----------------------------\n', genes)
    
            # Create gene nodes and relationships
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name:  # Only create node if gene name exists
                    session.run(
                        "MERGE (g:Gene {id: $gene_id, name: $gene_name, symbol: $gene_symbol})",
                        gene_id=gene_id, gene_name=gene_name, gene_symbol = gene_symbol
                    )
                    session.run(
                        "MATCH (p:Pathway {stId: $stId}), (g:Gene {id: $gene_id}) "
                        "MERGE (p)-[:INVOLVES]->(g)",
                        stId=stId, gene_id=gene_id
                    )
                
                session.run(
                    "MATCH (p:Pathway {stId: $stId}), (g:Gene {id: $gene_id}) "
                    "MERGE (p)-[:INVOLVES]->(g)",
                    stId=stId, gene_id=gene_id
                )
                
        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    
    finally:
        session.close()
        driver.close()

def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, stids, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=stids, yticklabels=stids,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(stids)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="Pathway-pathway similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path)
    ##plt.show()
    plt.close()
    
def create_pathway_map(reactome_file, output_file):
    """
    Extracts gene IDs with the same pathway STID and saves them to a new CSV file.

    Parameters:
    reactome_file (str): Path to the NCBI2Reactome.csv file.
    output_file (str): Path to save the output CSV file.
    """
    pathway_map = {}  # Dictionary to store gene IDs for each pathway STID

    # Read the NCBI2Reactome.csv file and populate the pathway_map
    with open(reactome_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            gene_id = row[0]
            pathway_stid = row[1]
            pathway_map.setdefault(pathway_stid, []).append(gene_id)

    # Write the pathway_map to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pathway STID", "Gene IDs"])  # Write header
        for pathway_stid, gene_ids in pathway_map.items():
            writer.writerow([pathway_stid, ",".join(gene_ids)])
    
    return pathway_map
        
def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def create_heatmap_with_stid(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def calculate_cluster_labels(net, dataloader, device, num_clusters=20):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels

def visualize_embeddings_pca(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
        
def visualize_embeddings_tsne(embeddings, cluster_labels, stid_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()

def save_significant_pathways_(clusters_with_significant_stId, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping):
    # Gather significant pathways and their connected genes
    significant_pathways_in_clusters = {}
    common_genes_in_clusters = {}  # To store common genes for each cluster

    for cluster_id, stIds in clusters_with_significant_stId.items():
        pathway_details = {}
        all_connected_genes = []

        if isinstance(stIds, list):
            # Handle the case where stIds is a list
            for single_stId in stIds:
                connected_genes = []
                genes = pathway_map.get(single_stId, [])
                for gene_id in genes:
                    gene_name = gene_id_to_name_mapping.get(gene_id, None)
                    gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                    if gene_name and gene_symbol:  # Ensure that both name and symbol exist
                        connected_genes.append({
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "gene_symbol": gene_symbol
                        })
                pathway_details[single_stId] = connected_genes
                all_connected_genes.append(set(gene['gene_symbol'] for gene in connected_genes))
        else:
            # Handle the case where stIds is a single value
            connected_genes = []
            genes = pathway_map.get(stIds, [])
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name and gene_symbol:  # Ensure that both name and symbol exist
                    connected_genes.append({
                        "gene_id": gene_id,
                        "gene_name": gene_name,
                        "gene_symbol": gene_symbol
                    })
            pathway_details[stIds] = connected_genes
            all_connected_genes.append(set(gene['gene_symbol'] for gene in connected_genes))
        
        significant_pathways_in_clusters[str(cluster_id)] = pathway_details

    # Save the significant pathways with connected genes to a JSON file
    output_dir = 'embedding/results/node_embeddings'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'significant_pathways_in_clusters_with_connected_genes.json')

    with open(output_file, 'w') as f:
        json.dump(significant_pathways_in_clusters, f, indent=4)

if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
