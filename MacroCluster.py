import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd


class MacroCluster:
    def __init__(self, seed_clusters, min_similarity, dynamic_addition=False, min_dynamic_update_similarity=0.9, dynamic_addition_mode='grand_centroid'):
        self.seed_clusters = seed_clusters
        self.seed_clusters_set = set(seed_clusters)
        self.prev_grand_centroid = None
        self.current_grand_centroid = None
        self.min_similarity = min_similarity
        self.dynamic_addition = dynamic_addition
        self.min_dynamic_update_similarity = min_dynamic_update_similarity
        self.initialized = False
        self.prev_clusters = None
        self.current_clusters = {}
        self.dynamic_addition_mode = dynamic_addition_mode
        # self.prev_clusters_set = None
        # self.current_clusters_set = {}

    def update_centroid(self, df, centroid_column):
        # Retrieve the centroids for the seed clusters
        if self.dynamic_addition and self.initialized:
            if self.dynamic_addition_mode == 'grand_centroid':
                new_seeds = self.get_new_seeds(df, centroid_column)
            else:
                new_seeds = self.get_new_seeds_bfs(df, centroid_column)

            print(f"New Seeds: {new_seeds}")
            self.seed_clusters.extend(new_seeds)
            self.seed_clusters_set = set(self.seed_clusters)

        centroids = df[df['Cluster'].isin(self.seed_clusters)][centroid_column].to_list()
        self.initialized = True

        # Update prev_grand_centroid and calculate new current_grand_centroid
        self.prev_grand_centroid = self.current_grand_centroid
        self.current_grand_centroid = np.mean(np.stack(centroids), axis=0)

    def print_centroid_difference(self):
        if self.prev_grand_centroid is not None and self.current_grand_centroid is not None:
            distance = cosine(self.prev_grand_centroid, self.current_grand_centroid)
            print(f"Cosine distance between previous and current grand centroid: {distance:.4f}")
        else:
            print("Previous or current grand centroid is not available.")

    def get_new_seeds_bfs(self, df, centroid_column):
        similar_clusters = set()

        # Iterate over each seed cluster's centroid
        for seed_cluster in self.seed_clusters:
            seed_centroid = df.loc[df['Cluster'] == seed_cluster, centroid_column].iloc[0]

            # Compare with every other cluster
            for cluster in df['Cluster'].unique():
                if cluster not in self.seed_clusters_set:
                    cluster_centroid = df.loc[df['Cluster'] == cluster, centroid_column].iloc[0]
                    similarity = 1 - cosine(seed_centroid, cluster_centroid)

                    # Add cluster to the set if it meets the similarity threshold
                    if similarity >= self.min_dynamic_update_similarity:
                        similar_clusters.add(cluster)

        # Print each cluster and its cosine similarity
        for cluster in similar_clusters:
            cluster_centroid = df.loc[df['Cluster'] == cluster, centroid_column].iloc[0]
            similarities = [1 - cosine(seed_centroid, cluster_centroid) for seed_centroid in
                            df.loc[df['Cluster'].isin(self.seed_clusters), centroid_column]]
            max_similarity = max(similarities)
            print(f"Cluster {cluster}: Highest Cosine Similarity = {max_similarity:.4f}")

        return list(similar_clusters)


    def get_all_clusters(self, df, centroid_column):
        # Calculate cosine similarity with the current_grand_centroid for each cluster
        cluster_similarities = {}
        for cluster in df['Cluster'].unique():
            cluster_centroid = df.loc[df['Cluster'] == cluster, centroid_column].iloc[0]
            similarity = 1 - cosine(self.current_grand_centroid, cluster_centroid)
            if similarity >= self.min_similarity or cluster in self.seed_clusters_set:
                cluster_similarities[cluster] = similarity

        # Sort clusters by similarity
        sorted_clusters = sorted(cluster_similarities.items(), key=lambda x: x[1], reverse=True)

        # Print each cluster and its cosine similarity
        cluster_numbers = []
        for cluster, similarity in sorted_clusters:
            if cluster in self.seed_clusters_set:
                print(f"**Cluster {cluster}: Cosine Similarity = {similarity:.4f}")
            else:
                print(f"Cluster {cluster}: Cosine Similarity = {similarity:.4f}")
                cluster_numbers.append(cluster)

        new_current_clusters = {}
        for cluster, similarity in sorted_clusters:
            point_count = df[df['Cluster'] == cluster].shape[0]
            # centroid = df[df['Cluster'] == cluster]
            cluster_centroid = df.loc[df['Cluster'] == cluster, centroid_column].iloc[0]
            new_current_clusters[cluster] = {'similarity': similarity, 'point_count': point_count,
                                            'centroid': cluster_centroid}
        # Update prev_clusters and current_clusters
        self.prev_clusters = self.current_clusters
        self.current_clusters = new_current_clusters

        return cluster_numbers

    def get_new_seeds(self, df, centroid_column):
        # Calculate cosine similarity with the current_grand_centroid for each cluster
        cluster_similarities = {}
        for cluster in df['Cluster'].unique():
            cluster_centroid = df.loc[df['Cluster'] == cluster, centroid_column].iloc[0]
            similarity = 1 - cosine(self.current_grand_centroid, cluster_centroid)
            # print(f'similarity: {cluster} - {similarity}')
            if similarity >= self.min_dynamic_update_similarity:
                cluster_similarities[cluster] = similarity

        # Sort clusters by similarity
        sorted_clusters = sorted(cluster_similarities.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_clusters)

        # Print each cluster and its cosine similarity
        cluster_numbers = []
        for cluster, similarity in sorted_clusters:
            if cluster not in self.seed_clusters_set:
                print(f'new cluster! {cluster}')
                cluster_numbers.append(cluster)

        return cluster_numbers


    def analyze_cluster_changes(self, top_n):
        if not self.prev_clusters or not self.current_clusters:
            print("Previous or current cluster data is not available.")
            return

        # Calculate growth and shrinkage for each cluster, excluding new clusters (infinity growth)
        growth_shrinkage_data = []
        new_clusters = []
        for cluster, current_data in self.current_clusters.items():
            prev_data = self.prev_clusters.get(cluster)
            if prev_data:
                point_growth = current_data['point_count'] - prev_data['point_count']
                percent_growth = (point_growth / prev_data['point_count']) * 100 if prev_data['point_count'] > 0 else 0
                growth_shrinkage_data.append((cluster, point_growth, percent_growth))
            else:
                new_clusters.append((cluster, current_data['point_count']))

        # Sort by percentage growth and shrinkage
        top_growing = sorted(growth_shrinkage_data, key=lambda x: x[2], reverse=True)[:top_n]
        top_shrinking = sorted(growth_shrinkage_data, key=lambda x: x[2])[:top_n]
        top_new_clusters = sorted(new_clusters, key=lambda x: x[1], reverse=True)[:top_n]

        # Print top growing and shrinking clusters
        print("Top Growing Clusters:")
        for cluster, growth, percent in top_growing:
            print(f"  Cluster {cluster}: Growth = {growth}, Percentage = {percent:.2f}%")

        print("Top Shrinking Clusters:")
        for cluster, shrinkage, percent in top_shrinking:
            print(f"  Cluster {cluster}: Shrinkage = {shrinkage}, Percentage = {percent:.2f}%")

        # Print top new clusters
        print("Top New Clusters:")
        for cluster, point_count in top_new_clusters:
            print(f"  Cluster {cluster}: New Points = {point_count}")
        # Total clusters and points growth
        total_prev_points = sum(data['point_count'] for data in self.prev_clusters.values())
        total_current_points = sum(data['point_count'] for data in self.current_clusters.values())
        total_cluster_growth = len(self.current_clusters) - len(self.prev_clusters)
        percent_cluster_growth = (total_cluster_growth / len(self.prev_clusters)) * 100 if len(self.prev_clusters) > 0 else float('inf')
        percent_points_growth = ((total_current_points - total_prev_points) / total_prev_points) * 100 if total_prev_points > 0 else float('inf')

        print(f"Total Cluster Growth: {total_cluster_growth} clusters, Percentage = {percent_cluster_growth:.2f}%")
        print(f"Total Points Growth: {total_current_points - total_prev_points} points, Percentage = {percent_points_growth:.2f}%")

        # Top clusters in terms of raw number of points
        top_clusters_raw_points = sorted(self.current_clusters.items(), key=lambda x: x[1]['point_count'], reverse=True)[:top_n]
        print("Top Clusters by Raw Number of Points:")
        for cluster, data in top_clusters_raw_points:
            print(f"  Cluster {cluster}: Points = {data['point_count']}")


    def summarize_current_state(self, top_n_dominant_clusters):
        if not self.current_clusters:
            print("Current cluster data is not available.")
            return

        # Total number of points
        total_points = sum(data['point_count'] for data in self.current_clusters.values())

        # Total number of clusters
        total_clusters = len(self.current_clusters)

        # Most dominant clusters by point count
        dominant_clusters = sorted(self.current_clusters.items(), key=lambda x: x[1]['point_count'], reverse=True)[:top_n_dominant_clusters]

        # Print summary
        print(f"Total Number of Points: {total_points}")
        print(f"Total Number of Clusters: {total_clusters}")
        print("Most Dominant Clusters:")
        most_dominant_clusters = []
        for cluster, data in dominant_clusters:
            print(f"  Cluster {cluster}: Points = {data['point_count']} - {(data['point_count']/total_points)*100:.3f}%")
            most_dominant_clusters.append(cluster)
        return most_dominant_clusters


if __name__ == "__main__":
    # Example usage:
    df = pd.DataFrame({
        'Cluster': [1, 1, 2, 2, 3],
        'centroid_column': [np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([0.2, 0.3]), np.array([0.2, 0.3]), np.array([0.3, 0.4])]
    })
    seed_clusters = [1, 2]
    macro_cluster = MacroCluster(seed_clusters, min_similarity=0.8)
    macro_cluster.update_centroid(df, 'centroid_column')
    macro_cluster.print_centroid_difference()
    