import numpy as np
import os
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist, cosine, euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import t
from sklearn.metrics import silhouette_samples, silhouette_score
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import squareform


class OnlineAgglomerative:
    def __init__(self, similarity_threshold, distance_type='cosine', cluster_combination_metric='f_score'):
        self.similarity_threshold = similarity_threshold
        self.distance_type = distance_type
        self.verbose = False
        self.clusters = []
        self.centroids = []
        self.silhouette_scores = []
        self.outliers = []
        self.cluster_labels = []
        self.cluster_radii = []
        self.cluster_point_counts = []
        self.farthest_centroid_distances = []
        self.threshold_cluster = 5
        self._n_threads = os.cpu_count()
        self.cluster_history = {}  # Format: {cluster_id: {'created': timestep, 'consumed': [(cluster_id, timestep)]}}
        self.consumed_clusters = {}  # Format: {consumed_cluster_id: (merged_into_cluster_id, timestep)}
        self.history_window = 4
        self.centroid_history = {}  # Format: {cluster_id: [centroid_timestep_1, centroid_timestep_2, ...]}
        self.radius_history = {}
        self.cluster_metrics = {}
        self.cluster_combination_metric = cluster_combination_metric

        self.current_timestep = 0
        self.filtered_clusters = set()





    def _mini_fit(self, embeddings):
        """
        Performs a mini clustering operation on a subset of embeddings; this is typically used for processing a batch of new data
        or outliers. This method applies hierarchical clustering to these embeddings based on similarity measures.

        Steps:
        1. Calculates a similarity matrix for the given embeddings to measure the similarity between each pair of embeddings.
        2. Converts the similarity matrix into a distance matrix (1 - similarity) to use distance-based clustering.
        3. Applies hierarchical clustering on the distance matrix using an average linkage method, grouping embeddings into clusters based on a defined distance threshold related to the similarity threshold.
        4. For each newly formed cluster, calculates the centroid by averaging the embeddings within the cluster.
        5. Updates the cluster's statistics, including the centroid, maximum distance from the centroid (radius), and the number of points within the cluster (this information is used later).


        Parameters:
        - embeddings: A subset of embeddings to be clustered, this is typically outliers.

        Returns:
        - clusters: The cluster assignments for the provided embeddings.
        - new_centroids: The centroids of the newly formed clusters.
        """
        # Calculate cosine similarity and distance matrix
        distance_matrix = None
        if self.distance_type == 'cosine':
          cosine_sim_matrix = cosine_similarity(embeddings)
          distance_matrix = np.clip(1 - cosine_sim_matrix, 0, None)
        if self.distance_type == 'euclidean':
          euclidean_dist_matrix = euclidean_distances(embeddings)
          distance_matrix = euclidean_dist_matrix


        similarity_threshold = self.similarity_threshold
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering
        Z = linkage(squareform(distance_matrix), method='average')
        distance_threshold = 1 - similarity_threshold
        clusters = fcluster(Z, t=distance_threshold, criterion='distance')

        # Initialize variables for new clusters
        new_cluster_labels = np.unique(clusters)
        new_centroids = []
        new_cluster_radii = []
        new_cluster_point_counts = []
        new_farthest_centroid_distances = []

        # Compute the centroid of each new cluster and other statistics
        for label in new_cluster_labels:
            cluster_points = embeddings[clusters == label]
            centroid = cluster_points.mean(axis=0)
            new_centroids.append(centroid)

            # Calculate distances from the centroid to all points in the cluster
            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            # Update the farthest distance, radius, and count for each new cluster
            new_farthest_centroid_distances.append(distances.max())
            new_cluster_radii.append(distances.max())
            new_cluster_point_counts.append(len(cluster_points))

        # Append the new cluster information to the existing attributes
        max_existing_label = max(self.cluster_labels, default=-1)
        new_cluster_ids = [i + max_existing_label + 1 for i in new_cluster_labels]

        self.centroids = np.concatenate((self.centroids, new_centroids))
        self.cluster_labels = np.concatenate((self.cluster_labels, new_cluster_ids))
        self.cluster_radii = np.concatenate((self.cluster_radii, np.array(new_cluster_radii)))
        self.cluster_point_counts = np.concatenate((self.cluster_point_counts, np.array(new_cluster_point_counts)))
        self.farthest_centroid_distances = np.concatenate((self.farthest_centroid_distances, np.array(new_farthest_centroid_distances)))
        return clusters, new_centroids



    def incremental_fit(self, new_embeddings, outlier_threshold, batch_size=100):
        """
        Processes new embeddings incrementally in specified batch sizes to update and refine the clustering model over time.

        Steps:
        1. Divides the incoming embeddings into batches (we do this to manage memory usage and computational load efficiently).
        2. For each batch, we asses the embeddings individually to determine if they are outliers based on the specified threshold.
        3. Non-outlier embeddings are integrated directly into the nearest existing cluster, then we update the cluster's centroid, radius, and point count.
        4. Outliers are collected until their number reaches the outlier threshold, at which point they are clustered together using a mini fit process -- this performs hierarchical clustering and
        potentially forms new clusters or allows them to be absorded into existing clusters.
        5. After batches, clusters may be combined to refine the clustering model further.

        Parameters:
        - new_embeddings: The new data points to be clustered.
        - outlier_threshold: The number of outliers that triggers an attempt to cluster these outliers.
        - batch_size: The number of embeddings processed in a single batch.

        Returns:
        None
        """

        start_time = time.time()
        outlier_absorption = False
        cluster_combination = False


        # Process embeddings in batches
        for i in range(0, len(new_embeddings), batch_size):
            batch_embeddings = new_embeddings[i:i + batch_size]

            current_batch_number = i // batch_size + 1

            # Calculate total number of batches
            # Adding 1 to account for the last batch which might be smaller
            total_batches = (len(new_embeddings) + batch_size - 1) // batch_size

            print(f'Processing batch {current_batch_number} of {total_batches}')
            for x_insert in tqdm(batch_embeddings):
                is_outlier, nearest_cluster_idx = self.is_outlier(x_insert)

                if is_outlier:
                    # print('is outlier!')
                    self.outliers.append(x_insert)

                    if len(self.outliers) >= outlier_threshold:
                        outliers_array = np.array(self.outliers)
                        new_clusters, new_centroids = self._mini_fit(outliers_array)
                        self.outliers = []
                        cluster_combination = True
                else:
                    self._update_cluster_on_insert(x_insert, nearest_cluster_idx)

            # Update model at the end of each batch
            if cluster_combination:
                print('Combining Clusters...')
                try:
                    if self.combine_clusters():
                        cluster_combination = False
                except Exception as e:
                    print(f'Error combining clusters: {e}')

            # Optional: Absorb outliers after each batch if needed
            if outlier_absorption:
                print('Absorbing outliers...')
                try:
                    self.absorb_outliers()
                    outlier_absorption = False
                except Exception as e:
                    print(f'Error absorbing outliers: {e}')

        elapsed_time = time.time() - start_time

        for i, label in enumerate(self.cluster_labels):
            try:
                # Initialize history if not already present
                if label not in self.centroid_history:
                    self.centroid_history[label] = [(self.centroids[i], self.current_timestep)]
                else:
                    # Append to existing history
                    self.centroid_history[label].append((self.centroids[i], self.current_timestep))

                if label not in self.radius_history:
                    self.radius_history[label] = [(self.cluster_radii[i], self.current_timestep)]
                else:
                    # Append to existing history
                    self.radius_history[label].append((self.cluster_radii[i], self.current_timestep))
            except:
                pass

        self.calculate_cluster_metrics()
        self.current_timestep += 1
        print(f"Incremental clustering with batch processing took {elapsed_time:.2f} seconds")



    def fit(self, embeddings):
        """
        Performs the initial clustering on the entire set of embeddings using hierarchical clustering. This sets up
        the initial clustering model, establishing a baseline for future incremental updates.

        Steps:
        1. Computes a similarity matrix for the embeddings to evaluate the similarity between each pair.
        2. Transforms the similarity matrix into a distance matrix (1 - similarity) to apply distance-based clustering.
        3. Utilizes hierarchical clustering with an 'average' linkage method on the distance matrix, forming initial clusters based on a distance threshold that inversely relates to the similarity threshold.
        4. Calculates the centroid for each cluster by averaging the embeddings within the cluster.
        5. Determines the radius (maximum distance from the centroid), point count, and farthest centroid distance for each cluster to support outlier detection and cluster updates in future processing.
        6. Optionally calculates silhouette scores for each sample within the clusters to evaluate the clustering quality, which can inform subsequent clustering adjustments or model tuning.

        This foundational clustering provides a comprehensive view of the data's initial structure, allowing for incremental refinement as new data is introduced or as the data distribution evolves over time.

        Parameters:
        - embeddings: The complete set of embeddings to be clustered initially.

        Returns:
        None, but establishes the model's initial clustering structure, including centroids, cluster labels, and related metrics.
        """


        start_time = time.time()

        # Step 2: Hierarchical clustering

        distance_matrix = None
        if self.distance_type == 'cosine':
          cosine_sim_matrix = cosine_similarity(embeddings)
          distance_matrix = np.clip(1 - cosine_sim_matrix, 0, None)
        if self.distance_type == 'euclidean':
          euclidean_dist_matrix = euclidean_distances(embeddings)
          distance_matrix = euclidean_dist_matrix


        similarity_threshold = self.similarity_threshold
        np.fill_diagonal(distance_matrix, 0)

        # Perform hierarchical clustering
        Z = linkage(squareform(distance_matrix), method='average')
        distance_threshold = 1 - similarity_threshold
        clusters = fcluster(Z, t=distance_threshold, criterion='distance')
        self.clusters = clusters

        # Initialize lists to store cluster information
        self.cluster_labels = np.unique(clusters)
        self.cluster_radii = []
        self.cluster_point_counts = []
        self.farthest_centroid_distances = []

        # Compute the centroid of each cluster and other statistics
        for label in self.cluster_labels:
            cluster_points = embeddings[clusters == label]
            centroid = cluster_points.mean(axis=0)
            self.centroids.append(centroid)

            # Calculate distances from the centroid to all points in the cluster
            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            # Update the farthest distance, radius, and count for each cluster
            self.farthest_centroid_distances.append(distances.max())
            self.cluster_radii.append(distances.max())
            self.cluster_point_counts.append(len(cluster_points))

        # Compute the silhouette scores for each sample
        self.silhouette_scores = silhouette_samples(distance_matrix, clusters, metric="precomputed")

        elapsed_time = time.time() - start_time
        print(f"Hierarchical clustering took {elapsed_time:.2f} seconds")


        for i, label in enumerate(self.cluster_labels):
            # Initialize history if not already present
            if label not in self.centroid_history:
                self.centroid_history[label] = [(self.centroids[i], self.current_timestep)]
            else:
                # Append to existing history
                self.centroid_history[label].append((self.centroids[i], self.current_timestep))

            if label not in self.radius_history:
                self.radius_history[label] = [(self.cluster_radii[i], self.current_timestep)]
            else:
                # Append to existing history
                self.radius_history[label].append((self.cluster_radii[i], self.current_timestep))

        self.calculate_cluster_metrics()
        self.current_timestep += 1



    def calculate_cluster_metrics(self):
        """
        Calculate and store cluster metrics based on their centroid and radius history.

        Iterates through each cluster's historical data to calculate changes in the centroid's position
        and the cluster's radius over the last two timesteps. Metrics calculated include the Euclidean
        distance between the last two centroids and the percentage change in radius.
        Metrics are stored as a dictionary within the cluster_metrics attribute of the object.
        Parameters:
        - None.

        Returns:
        - None.
        """

        for cluster_id in self.cluster_labels:
            centroid_history = [item[0] for item in self.centroid_history.get(cluster_id, [])]
            radius_history = [item[0] for item in self.radius_history.get(cluster_id, [])]
            if len(centroid_history) < 2 or len(radius_history) < 2:
                continue  # Skip if not enough data
            # Calculate metrics
            centroid_distance = np.linalg.norm(centroid_history[-1] - centroid_history[-2])
            radius_change = radius_history[-1] - radius_history[-2]
            percent_radius_change = (radius_change / radius_history[-2] * 100) if radius_history[-2] != 0 else float('inf')

            # Store metrics in the object
            self.cluster_metrics[cluster_id] = {
                "timestep": self.current_timestep,
                "centroid_distance": centroid_distance,
                "radius_change": radius_change,
                "percent_radius_change": percent_radius_change
            }



    def predict_single_embedding(self, embedding):
        """
        Determines the closest centroid to a given embedding based on the configured distance metric (cosine or euclidean).
        We use this for assigning a cluster to a new or isolated data point after the model has been trained.

        Steps:
        1. Calculates the distance between the given embedding and each of the existing cluster centroids using the specified distance metric.
        2. Identifies the nearest centroid by selecting the one with the smallest distance to the embedding.
        3. Returns the label of the nearest cluster along with the centroid vector itself.


        Parameters:
        - embedding: A single data point's embedding vector to classify into an existing cluster.

        Returns:
        - A tuple containing the label of the nearest cluster and the vector of the closest centroid.
        """
        # Calculate the distance to each cluster centroid
        if self.distance_type == 'cosine':
            distances = np.array([cosine(embedding, centroid) for centroid in self.centroids])
        elif self.distance_type == 'euclidean':
            distances = np.linalg.norm(self.centroids - embedding, axis=1)
        else:
            raise ValueError("Unsupported distance type. Use 'cosine' or 'euclidean'.")

        # Find the index of the nearest centroid
        nearest_centroid_idx = np.argmin(distances)
        return self.cluster_labels[nearest_centroid_idx], self.centroids[nearest_centroid_idx]

    def predict(self, X):
        """
        Predicts cluster labels and the closest centroids for a given set of embeddings (X).
        We use multithreading here to compute labels in parallel for large datasets.

        Steps:
        1. Computes similarities between the embeddings and existing cluster centroids.
        2. Transforms these similarities into distances (1 - similarity), as closer points have smaller distances.
        3. Identifies the closest centroid for each embedding based on the minimum distance (maximum similarity).
        4. Utilizes a ThreadPoolExecutor to distribute the computation across multiple threads, each handling a subset of the embeddings.
        5. Aggregates the results from all threads, forming a complete set of labels and closest centroids for the input data.

        Returns:
        - clusters: An array of the predicted cluster labels for each embedding.
        - closest_centroids: An array of the closest centroid vectors corresponding to each embedding's predicted cluster.
        """


        def compute_labels(start, end):
            # Calculate cosine similarity (values range from -1 to 1)
            similarities = cosine_similarity(X[start:end], self.centroids)

            # Convert similarities to distances (1 - similarity)
            # Closer points have smaller distances
            distances = 1 - similarities

            # Get the index of the centroid with the minimum distance (maximum similarity)
            return np.argmin(distances, axis=1)

        labels = np.empty(X.shape[0], dtype=int)
        closest_centroids = []
        clusters = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_threads) as executor:
            futures = []
            # batch_size = len(X) // self._n_threads
            batch_size = max(len(X) // self._n_threads, 1)

            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                futures.append(executor.submit(compute_labels, i, end))

            # Combine results from all threads
            for i, future in enumerate(futures):
                start = i * batch_size
                end = min(start + batch_size, len(X))
                labels[start:end] = future.result()

        for label in labels:
            closest_centroids.append(self.centroids[label])
            clusters.append(self.cluster_labels[label])


        return clusters, closest_centroids


    def _update_cluster_on_insert(self, x_insert, cluster_idx):
        """
        Updates the specified cluster with a new data point, recalculating the centroid, adjusting the cluster's radius,
        and incrementing the point count.

        Steps:
        1. Computes the new centroid by incorporating the new data point (x_insert) into the existing centroid calculation, accounting for the updated number of points within the cluster.
        2. Updates the internal representation of the centroid for the specified cluster to reflect the addition of the new point.
        3. Increments the point count for the cluster to include the new data point.
        4. Calculates the distance from the new centroid to the inserted data point. If this distance is greater than the current farthest distance recorded for the cluster, the farthest distance is updated to this new value, potentially adjusting the cluster's radius.
        5. If the cluster's history does not already contain an entry for the current timestep, it is updated to include the creation or modification of the cluster due to the insertion.

        Parameters:
        - x_insert: The embedding vector of the new data point to be inserted into the cluster.
        - cluster_idx: The index of the cluster being updated with the new data point.

        Returns:
        None, but modifies the model's state by updating the specified cluster's centroid, point count, and farthest distance metrics.
        """
        # Calculate the new centroid
        n_points = self.cluster_point_counts[cluster_idx]  # The current number of points in the cluster
        current_centroid = self.centroids[cluster_idx]
        new_centroid = (current_centroid * n_points + x_insert) / (n_points + 1)

        # Update the centroid in the class instance
        self.centroids[cluster_idx] = new_centroid

        # Update the number of points in the cluster
        self.cluster_point_counts[cluster_idx] = n_points + 1

        # Calculate the distance of the new point from the new centroid
        distance_to_new_point = np.linalg.norm(new_centroid - x_insert)

        # Update the farthest distance if the new point is farther than the current farthest
        if distance_to_new_point > self.farthest_centroid_distances[cluster_idx]:
            self.farthest_centroid_distances[cluster_idx] = distance_to_new_point

        if cluster_idx not in self.cluster_history:
            self.cluster_history[cluster_idx] = {'created': self.current_timestep, 'consumed': []}



    def is_outlier(self, x_insert, alpha=0.05):

        """
        Evaluates whether a given data point (x_insert) is an outlier relative to the existing clusters, based on its distance
        to the nearest cluster's centroid and the cluster's radius. Utilizes the Grubbs' test for clusters exceeding a certain
        size threshold, providing a statistical method to determine the significance of an observation being an outlier.

        We use Grubbs' statistic because of its ability to identify outliers in a dataset by comparing
        the most extreme values to the mean, under the assumption that the dataset follows a normal distribution.

        This method is generally understood to be robust in detecting a single outlier by measuring the largest deviation in terms of standard deviations from the mean.
        Within our purpose, by applying the Grubbs' test, we can more rigorously assess whether a data point's distance
        from the cluster centroid is statistically significant.
        Steps:
        1. Calculates the distance from the data point to each cluster's centroid using the configured distance metric.
        2. Identifies the nearest cluster to the data point and its corresponding radius.
        3. Initially checks if the data point falls within the nearest cluster's radius; if so, it is not considered an outlier.
        4. For clusters exceeding a certain size threshold, applies the Smirnov-Grubbs test to determine if the data point is
          an outlier based on a statistical comparison of its distance to the cluster centroid against the distribution of
          distances within the cluster.
        5. For smaller clusters or when the data point falls outside the nearest cluster's radius, employs a heuristic based on
          the farthest known point within the cluster to make a preliminary determination of outlier status.

        Parameters:
        - x_insert: The embedding vector of the data point to be assessed.
        - alpha: The significance level used for the Smirnov-Grubbs test, controlling the test's strictness.

        Returns:
        - A tuple (is_outlier, nearest_cluster_idx), where is_outlier is a boolean indicating whether the data point is considered an outlier, and nearest_cluster_idx is the index of the nearest cluster.
        """

        # Calculate the distance from x_insert to each cluster center
        distances_to_centroids = cdist([x_insert], self.centroids, metric=self.distance_type).flatten()

        # Identify the nearest cluster
        nearest_cluster_idx = np.argmin(distances_to_centroids)
        nearest_cluster_radius = self.cluster_radii[nearest_cluster_idx]
        nearest_cluster_distance = distances_to_centroids[nearest_cluster_idx]


        # Check if the point is within the nearest cluster's radius
        if nearest_cluster_distance <= nearest_cluster_radius:
            # The point is not an outlier
            if self.verbose:
              print(f'Within a radius: {nearest_cluster_distance:.3f} versus {nearest_cluster_radius:.3f}')
            return False, nearest_cluster_idx



        # Use the point count for the nearest cluster instead of points list
        n_points = self.cluster_point_counts[nearest_cluster_idx]

        # Determine if the cluster size is above the threshold to use the Smirnov-Grubbs test
        self.threshold_cluster = 10
        if n_points >= self.threshold_cluster:
            # print(f'n points: {n_points} > {self.threshold_cluster}')
            # Perform the Smirnov-Grubbs test
            mean_distance = np.mean(distances_to_centroids)
            std_distance = np.std(distances_to_centroids)
            t_value = t.ppf(1 - alpha/(2*n_points), df=n_points-2)
            grubbs_threshold = ((n_points - 1) * t_value) / np.sqrt(n_points * (n_points - 2 + t_value**2))

            grubbs_statistic = abs(nearest_cluster_distance - mean_distance) / std_distance
            if self.verbose:
              print(f'grubbs stat: {grubbs_statistic} versus threshold: {grubbs_threshold}')

            if grubbs_statistic > grubbs_threshold:
                # The point is an outlier
                return True, nearest_cluster_idx
        else:
            # If the cluster size is below the threshold, use the alternative method
            # Use the farthest distance within the cluster to estimate the mean distance
            # print(f'not over threshold cluster: {n_points} versus {self.threshold_cluster}')
            mean_distance_within_cluster = self.farthest_centroid_distances[nearest_cluster_idx] / 2  # Estimate as half of the farthest distance
            if self.verbose:
              print(f'nearest_cluster_distance: {nearest_cluster_distance} versus {nearest_cluster_radius + mean_distance_within_cluster}')
            if nearest_cluster_distance > nearest_cluster_radius + mean_distance_within_cluster:
                return True, nearest_cluster_idx

        # The point is not an outlier
        return False, nearest_cluster_idx

    def absorb_outliers(self):
        """
        Attempts to absorb identified outliers into existing clusters if they fall within the radius of any cluster. This
        method iterates through each outlier and calculates its distance to all cluster centroids, determining if it is
        sufficiently close to be considered part of a cluster rather than remaining classified as an outlier.

        The rationale behind absorbing outliers is to refine the cluster boundaries by incorporating points that, upon
        reassessment, can be reasonably assigned to a nearby cluster. This process helps in dynamically adjusting the model
        to better fit the data, reducing the number of points considered anomalous and potentially uncovering nuanced patterns
        within the dataset.

        Steps:
        1. For each outlier, compute its distance to every cluster centroid to find the nearest cluster.
        2. If an outlier's distance to the nearest cluster centroid is less than or equal to the cluster's radius, consider the
          outlier as being within the cluster's boundary and absorb it into the cluster.
        3. Update the cluster's statistics (e.g., centroid, radius) to reflect the inclusion of the outlier.
        4. Remove absorbed outliers from the list of outliers, leaving only those that could not be integrated into a cluster.

        Returns:
        None, but updates the internal state by potentially reducing the number of outliers and adjusting clusters to include newly absorbed points.
        """

        # Iterate over the outliers to determine if they can be absorbed by a cluster
        for outlier in self.outliers:
            # Calculate the distance from the outlier to each cluster center
            distances = [self._calculate_distance(outlier, centroid) for centroid in self.centroids]

            # Find the nearest cluster
            nearest_cluster_idx = np.argmin(distances)
            distance_to_nearest = distances[nearest_cluster_idx]
            nearest_cluster_radius = self.cluster_radii[nearest_cluster_idx]

            # Check if the outlier is within the cluster's radius
            if distance_to_nearest <= nearest_cluster_radius:
                # If it is within the radius, it is absorbed
                self._update_cluster_absorb_outliers(nearest_cluster_idx, outlier)
            else:
                # If not, it remains an outlier and will be assessed again later
                continue

        # Clear the outliers that have been absorbed
        self.outliers = [outlier for outlier in self.outliers if not self._is_absorbed(outlier)]

    def _calculate_distance(self, point, centroid):

        """
        Calculates the distance between a given point and a centroid using the configured distance metric (cosine or euclidean).
        Parameters:
        - point: The data point's embedding vector.
        - centroid: The centroid's embedding vector against which the distance is calculated.

        Returns:
        - The distance between the point and the centroid as a float.
        """
        if self.distance_type == 'cosine':
            return cosine(point, centroid)
        elif self.distance_type == 'euclidean':
            return distance.euclidean(point, centroid)

    def _is_absorbed(self, point):
        """
        Determines if a given point has been absorbed into a cluster based on its distance to the nearest cluster's centroid.
        This method is part of the process to reassess outliers and integrate them into existing clusters if appropriate.

        Steps:
        1. Calculates the distance from the point to each cluster's centroid.
        2. Identifies the nearest cluster based on these distances.
        3. Checks if the point's distance to the nearest cluster is less than or equal to the cluster's radius.

        Returns:
        - A boolean indicating whether the point has been absorbed into the nearest cluster.
        """
        # Calculate the distance to each cluster to check if a point has been absorbed
        distances = [self._calculate_distance(point, centroid) for centroid in self.centroids]
        nearest_cluster_idx = np.argmin(distances)
        return distances[nearest_cluster_idx] <= self.cluster_radii[nearest_cluster_idx]

    def _update_cluster_absorb_outliers(self, cluster_idx, point):
        """
        Integrates an outlier into a specified cluster, updating the cluster's centroid, radius, and point count to reflect
        the inclusion of the new point.

        Steps:
        1. Computes the new centroid by including the outlier in the existing centroid calculation, taking into account the updated number of points within the cluster.
        2. Updates the cluster's internal representation to include the new point, recalculating the centroid, adjusting the radius if necessary, and incrementing the point count.
        3. Reassesses the cluster's farthest point distance to ensure the radius accurately represents the spatial extent of the cluster.

        Parameters:
        - cluster_idx: The index of the cluster being updated.
        - point: The embedding vector of the outlier being absorbed.

        Returns:
        None, but modifies the cluster's centroid, radius, and point count to include the absorbed outlier.
        """
        # Calculate the new centroid
        n_points = self.cluster_point_counts[cluster_idx]  # The current number of points in the cluster
        current_centroid = self.centroids[cluster_idx]
        new_centroid = (current_centroid * n_points + point) / (n_points + 1)

        # Update the centroid in the class instance
        self.centroids[cluster_idx] = new_centroid

        # Update the number of points in the cluster
        self.cluster_point_counts[cluster_idx] = n_points + 1

        # Calculate the distance of the new point from the new centroid
        distance_to_new_point = np.linalg.norm(new_centroid - point)

        # Update the farthest distance if the new point is farther than the current farthest
        self.farthest_centroid_distances[cluster_idx] = max(self.farthest_centroid_distances[cluster_idx], distance_to_new_point)

        # Update the cluster radius if necessary
        self.cluster_radii[cluster_idx] = max(self.cluster_radii[cluster_idx], distance_to_new_point)


    def try_combine_clusters(self, i, j):
        """
        Attempts to combine two clusters based on their similarity and distance metrics. It evaluates the potential
        combination by comparing the distance between centroids to the sum of their radii, adjusted for overlap. A successful
        combination results in a single, merged cluster with a new centroid, radius, and point count reflective of the union
        of the two original clusters.

        Parameters:
        - i, j: Indices of the two clusters being considered for combination.

        Returns:
        - A tuple containing the new F-index value if the combination is beneficial, along with the indices of the combined clusters, or None if the combination does not improve clustering.
        """

        cosine_dist = 1 - cosine_similarity([self.centroids[i]], [self.centroids[j]])[0, 0]
        overlapping_element = 2
        radii_sum = (self.cluster_radii[i] + self.cluster_radii[j]) * overlapping_element
        if cosine_dist > radii_sum:
            return None

        total_points = self.cluster_point_counts[i] + self.cluster_point_counts[j]
        combined_centroid = (self.centroids[i] * self.cluster_point_counts[i] +
                             self.centroids[j] * self.cluster_point_counts[j]) / total_points
        combined_radius = max(self.cluster_radii[i], self.cluster_radii[j])

        tentative_centroids = np.delete(self.centroids, j, axis=0)
        tentative_centroids[i] = combined_centroid
        tentative_radii = np.delete(self.cluster_radii, j, axis=0)
        tentative_radii[i] = combined_radius
        tentative_point_counts = np.delete(self.cluster_point_counts, j, axis=0)
        tentative_point_counts[i] = total_points

        new_f_index = self.calculate_pseudo_f_index(tentative_centroids, tentative_radii, tentative_point_counts)
        if new_f_index > self.calculate_pseudo_f_index():
            return new_f_index, (i, j)
        return None


    def process_batch(self, batch, best_f_index):
        """
        Processes a batch of cluster pairs to identify the best combination that improves the overall clustering quality, as
        measured by the F-index. Iterates through pairs of clusters within the batch, attempting to combine them and assessing
        the impact on the clustering structure.

        Parameters:
        - batch: A list of cluster indices to be considered for merging.
        - best_f_index: The current best F-index against which improvements are compared.

        Returns:
        - The updated best F-index and the pair of cluster indices representing the best combination found in this batch, or
          None if no beneficial combination is found.
        """
        best_combination = None
        # for i in batch:
        for i in tqdm(batch, desc="Processing batch"):
            for j in range(i + 1, len(self.centroids)):
                result = self.try_combine_clusters(i, j)
                if result and result[0] > best_f_index:
                    best_f_index, best_combination = result
        return best_f_index, best_combination



    def filter_centroid_pairs(self, similarity_threshold=0.8, top_n=100):
        """
        Filters pairs of centroids based on their similarity, retaining only those pairs that exceed a specified
        similarity threshold. This method prioritizes pairs with higher similarity, aiming to identify the most promising
        candidates for cluster combination.

        Parameters:
        - similarity_threshold: Minimum similarity required for a pair to be considered.
        - top_n: The number of top pairs to retain based on similarity.

        Returns:
        - A list of centroid pairs that meet or exceed the similarity threshold, limited to the top_n pairs if specified.
        """
        # Compute cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(self.centroids)

        # Identify pairs that meet the similarity threshold
        pairs_with_scores = []
        num_embeddings = len(self.centroids)
        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):
                sim_score = cosine_sim_matrix[i, j]
                if sim_score >= similarity_threshold:
                    pairs_with_scores.append(((i, j), sim_score))

        # Sort pairs by similarity score in descending order and select top N
        pairs_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs_with_scores[:top_n] if top_n is not None else pairs_with_scores

        # Extract just the pairs (without scores)
        top_pairs = [pair for pair, score in top_pairs]

        return top_pairs



    def shed_clusters(self, df, cluster_column, min_cluster_size, max_weeks_old):
        """
        Filters clusters based on their size and age, retaining only those that are sufficiently large or recently updated.
        This method aims to refine the cluster set by removing smaller, potentially less significant clusters and those not
        recently updated, which we found to often be comprised of noise.

        Parameters:
        - df: DataFrame containing cluster data.
        - cluster_column: The name of the column containing cluster labels.
        - min_cluster_size: Minimum size a cluster must have to be retained.
        - max_weeks_old: Maximum age (in weeks) of clusters to be considered for retention.

        Returns:
        - A filtered DataFrame containing only the data for clusters that meet the size and age criteria.
        """

        # Convert max_date string to datetime
        max_date = pd.to_datetime(df['date']).max()
        max_date = pd.to_datetime(max_date)

        # Determine the cutoff date
        cutoff_date = max_date - timedelta(weeks=max_weeks_old)

        # Separate old and new data
        old_data = df[pd.to_datetime(df['date']) <= cutoff_date]
        new_data = df[pd.to_datetime(df['date']) > cutoff_date]

        # Identify large clusters in old data
        old_cluster_sizes = old_data[cluster_column].value_counts()
        large_old_clusters = set(old_cluster_sizes[old_cluster_sizes >= min_cluster_size].index)

        # Combine large old clusters with all new cluster labels
        combined_clusters = large_old_clusters.union(set(new_data[cluster_column]))

        # Update the OnlineAgglomerative instance attributes
        combined_cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label in combined_clusters]
        self.centroids = [self.centroids[i] for i in combined_cluster_indices]
        self.cluster_labels = [self.cluster_labels[i] for i in combined_cluster_indices]
        self.cluster_radii = [self.cluster_radii[i] for i in combined_cluster_indices]
        self.cluster_point_counts = [self.cluster_point_counts[i] for i in combined_cluster_indices]
        self.farthest_centroid_distances = [self.farthest_centroid_distances[i] for i in combined_cluster_indices]

        # Combine filtered old data with all new data for the DataFrame
        final_filtered_df = pd.concat([old_data[old_data[cluster_column].isin(large_old_clusters)], new_data])

        # Print statistics
        original_cluster_count = df[cluster_column].nunique()
        # new_cluster_count = final_filtered_df[cluster_column].nunique()
        new_cluster_count = len(online_agglo.centroids)

        print("Original Number of Clusters:", original_cluster_count)
        print("New Number of Clusters:", new_cluster_count)

        # Return the combined DataFrame
        return final_filtered_df



    def combine_clusters(self):
        """
        Evaluates and executes the best possible combination of clusters to enhance the overall quality of the clustering
        structure. Utilizes a systematic approach to identify and merge clusters that are closely related, aiming to
        optimize the clustering configuration as reflected by an improvement in the F-index.

        Returns:
        - A boolean indicating whether any cluster combination was performed.
        """
        print('Combining clusters...')
        pairs_to_check = self.filter_centroid_pairs()

        num_pairs = len(pairs_to_check)

        unique_indices = set(index for pair in pairs_to_check for index in pair)

        # Determine batch size
        num_indices = len(unique_indices)
        batch_size = max(num_indices // self._n_threads, 1)
        batches = [list(unique_indices)[i:i + batch_size] for i in range(0, num_indices, batch_size)]

        if self.cluster_combination_metric == 'f_score':
          # Initialize variables to store the best combination
          best_f_index = self.calculate_pseudo_f_index()
          best_combination = None

          # Use ThreadPoolExecutor to process each batch
          for i, j in tqdm(pairs_to_check, desc="Checking for best combination..."):
              result = self.try_combine_clusters(i, j)
              if result and result[0] > best_f_index:
                  best_f_index, best_combination = result


          if best_combination is not None:
              print(f'best combination: {best_combination}')
              i, j = best_combination
              total_points = self.cluster_point_counts[i] + self.cluster_point_counts[j]

              self.cluster_labels[self.cluster_labels == j] = i
              self.cluster_labels[self.cluster_labels > j] -= 1

              self.centroids = np.delete(self.centroids, j, axis=0)
              self.cluster_radii = np.delete(self.cluster_radii, j, axis=0)
              self.cluster_point_counts = np.delete(self.cluster_point_counts, j, axis=0)

              self.centroids[i] = (self.centroids[i] * self.cluster_point_counts[i] +
                                  self.centroids[j] * self.cluster_point_counts[j]) / total_points

              self.cluster_radii[i] = max(self.cluster_radii[i], self.cluster_radii[j])
              self.cluster_point_counts[i] = total_points

              if i in self.cluster_history and j in self.cluster_history:
                  self.cluster_history[i]['consumed'].append((j, self.current_timestep))
                  self.consumed_clusters[j] = (i, self.current_timestep)

              return True  # Indicate that a combination occurred

          return False  # Indicate that no combination occurred

        elif self.cluster_combination_metric == 'silhouette':
          current_silhouette_score = self.calculate_silhouette_score()
          best_combination = None

          # Use ThreadPoolExecutor to process each batch
          for i, j in tqdm(pairs_to_check, desc="Checking for best combination..."):
              result = self.try_combine_clusters(i, j)
              if result and result[0] > current_silhouette_score:
                  current_silhouette_score, best_combination = result

          if best_combination is not None:
              print(f'best combination: {best_combination}')
              i, j = best_combination
              self.apply_cluster_combination(i, j)  # Apply the best found cluster combination
              return True  # Indicate that a combination occurred

          return False  # Indicate that no combination occurred
        else:
          return False

    def calculate_silhouette_score(self):
        """
        Calculates the silhouette score for the current clustering configuration using all cluster labels and data points.

        Returns:
        - The calculated silhouette score, indicating the relative quality of the clustering structure.
        """
        if len(set(self.cluster_labels)) <= 1:
            return -1  # Silhouette score not meaningful for a single cluster

        # Calculate the silhouette score for the clustering
        score = silhouette_score(self.data_points, self.cluster_labels, metric='euclidean')
        return score

    def calculate_pseudo_f_index(self, centroids=None, radii=None, point_counts=None):
        """
        Calculates the Pseudo F-index, a measure of clustering quality that balances the between-cluster variability against
        the within-cluster variability.

        Parameters (optional):
        - centroids: Centroids of the clusters. Uses existing centroids if not provided.
        - radii: Radii of the clusters. Uses existing radii if not provided.
        - point_counts: Number of points in each cluster. Uses existing counts if not provided.

        Returns:
        - The calculated Pseudo F-index value, indicating the relative quality of the clustering structure.
        """
        # print('Calculating Pseudo F-Index')
        # Use the instance variables if none are provided
        if centroids is None:
            centroids = self.centroids
        if radii is None:
            radii = self.cluster_radii
        if point_counts is None:
            point_counts = self.cluster_point_counts

        N = sum(point_counts)  # Total number of points
        k = len(centroids)  # Number of clusters

        if k <= 1 or N <= k:
            return 0

        # Grand centroid of all points
        weighted_centroids = np.array(centroids) * np.array(point_counts)[:, None]
        grand_centroid = np.sum(weighted_centroids, axis=0) / N

        # Between-group variability (SSB)
        centroid_diff = np.array(centroids) - grand_centroid
        SSB = np.sum(point_counts * np.einsum('ij,ij->i', centroid_diff, centroid_diff))

        # Within-group variability (SSW) using radius squared
        SSW = np.sum(point_counts * np.array(radii)**2)

        # Calculate the Pseudo F-index
        F = (SSB / (k - 1)) / (SSW / (N - k))
        # print('Finished calculating Pseudo F-Index')
        return F


if __name__ == "__main__":
    # Example usage:
    embeddings = np.random.rand(10, 5)  # Dummy data: 10 samples, 5 features each
    agglomerative = OnlineAgglomerative(similarity_threshold=0.8)
    agglomerative.fit(embeddings)
