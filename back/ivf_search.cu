// ivf_search.cu
// Búsqueda IVF (Inverted File Index) sobre embeddings, implementada en CUDA C++.
// Equivalente en C/CUDA de la función buscar_ivf_gpu (versión CuPy).
//
// Recibe:
//   - lista_embeddings: vector de vectores (dataset de embeddings)
//   - query_embedding:  vector de floats (el prompt/query embebido)
//   - lista_textos:     vector de strings (un texto por embedding)
// Devuelve:
//   - vector<string> con los top_k textos más relevantes

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cfloat>

// ---------------------------------------------------------------------
// Kernel 1: asigna cada vector del dataset al centroide más cercano
// ---------------------------------------------------------------------
__global__ void assign_labels_kernel(
    const float* dataset,    // [num_vectors x dim]
    const float* centroides, // [n_lists x dim]
    int* labels,             // [num_vectors]
    int num_vectors,
    int dim,
    int n_lists
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    float best_dist = FLT_MAX;
    int best_cluster = 0;

    for (int c = 0; c < n_lists; ++c) {
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = dataset[idx * dim + d] - centroides[c * dim + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_cluster = c;
        }
    }
    labels[idx] = best_cluster;
}

// ---------------------------------------------------------------------
// Kernel 2: acumula sumas y conteos por centroide (paso de actualización)
// ---------------------------------------------------------------------
__global__ void accumulate_centroids_kernel(
    const float* dataset,
    const int* labels,
    float* sums,   // [n_lists x dim], debe iniciarse en 0 antes de llamar
    int* counts,   // [n_lists], debe iniciarse en 0 antes de llamar
    int num_vectors,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;

    int cluster = labels[idx];
    atomicAdd(&counts[cluster], 1);
    for (int d = 0; d < dim; ++d) {
        atomicAdd(&sums[cluster * dim + d], dataset[idx * dim + d]);
    }
}

// ---------------------------------------------------------------------
// Kernel 3: divide sumas entre conteos -> nuevos centroides
// ---------------------------------------------------------------------
__global__ void update_centroids_kernel(
    float* centroides,
    const float* sums,
    const int* counts,
    int n_lists,
    int dim
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_lists) return;

    int count = counts[c];
    if (count > 0) {
        for (int d = 0; d < dim; ++d) {
            centroides[c * dim + d] = sums[c * dim + d] / count;
        }
    }
}

// ---------------------------------------------------------------------
// Kernel 4: distancia euclídea al cuadrado entre la query y cada
// elemento indicado en `indices` (los pertenecientes a los clusters
// seleccionados por n_probe). Misma firma que ya tenías.
// ---------------------------------------------------------------------
extern "C" __global__ void compute_distances(
    const float* query,
    const float* dataset,
    const int* indices,
    float* distances,
    int dim,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int dataset_idx = indices[idx];
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = query[d] - dataset[dataset_idx * dim + d];
            dist += diff * diff;
        }
        distances[idx] = dist;
    }
}

// ---------------------------------------------------------------------
// Función host: búsqueda IVF completa
// ---------------------------------------------------------------------
std::vector<std::string> buscar_ivf_gpu(
    const std::vector<std::vector<float>>& lista_embeddings,
    const std::vector<float>& query_embedding,
    const std::vector<std::string>& lista_textos,
    int n_lists = 4,
    int n_probe = 2,
    int top_k = 5,
    int kmeans_iters = 10
) {
    if (lista_embeddings.empty() || query_embedding.empty()) {
        return {};
    }

    int num_vectors = static_cast<int>(lista_embeddings.size());
    int dim = static_cast<int>(lista_embeddings[0].size());

    n_lists = std::min(n_lists, num_vectors);
    n_probe = std::min(n_probe, n_lists);

    // --- 1. Aplanar el dataset a un arreglo contiguo (row-major) ---
    std::vector<float> dataset_flat(static_cast<size_t>(num_vectors) * dim);
    for (int i = 0; i < num_vectors; ++i) {
        std::copy(lista_embeddings[i].begin(), lista_embeddings[i].end(),
                   dataset_flat.begin() + static_cast<size_t>(i) * dim);
    }

    // --- 2. Reservar memoria en GPU y copiar datos ---
    float *d_dataset, *d_query, *d_centroides, *d_sums;
    int *d_labels, *d_counts;

    cudaMalloc(&d_dataset, dataset_flat.size() * sizeof(float));
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_centroides, static_cast<size_t>(n_lists) * dim * sizeof(float));
    cudaMalloc(&d_sums, static_cast<size_t>(n_lists) * dim * sizeof(float));
    cudaMalloc(&d_labels, num_vectors * sizeof(int));
    cudaMalloc(&d_counts, n_lists * sizeof(int));

    cudaMemcpy(d_dataset, dataset_flat.data(), dataset_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query_embedding.data(), dim * sizeof(float), cudaMemcpyHostToDevice);

    // --- 3. Inicializar centroides con vectores aleatorios del dataset ---
    std::vector<int> all_idx(num_vectors);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all_idx.begin(), all_idx.end(), rng);

    std::vector<float> centroides_init(static_cast<size_t>(n_lists) * dim);
    for (int i = 0; i < n_lists; ++i) {
        int src = all_idx[i];
        std::copy(dataset_flat.begin() + static_cast<size_t>(src) * dim,
                   dataset_flat.begin() + static_cast<size_t>(src + 1) * dim,
                   centroides_init.begin() + static_cast<size_t>(i) * dim);
    }
    cudaMemcpy(d_centroides, centroides_init.data(), centroides_init.size() * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Iteraciones de K-Means en GPU ---
    int threads = 256;
    int blocks_vectors = (num_vectors + threads - 1) / threads;
    int blocks_lists = (n_lists + threads - 1) / threads;

    for (int iter = 0; iter < kmeans_iters; ++iter) {
        assign_labels_kernel<<<blocks_vectors, threads>>>(
            d_dataset, d_centroides, d_labels, num_vectors, dim, n_lists);

        cudaMemset(d_sums, 0, static_cast<size_t>(n_lists) * dim * sizeof(float));
        cudaMemset(d_counts, 0, n_lists * sizeof(int));

        accumulate_centroids_kernel<<<blocks_vectors, threads>>>(
            d_dataset, d_labels, d_sums, d_counts, num_vectors, dim);
        update_centroids_kernel<<<blocks_lists, threads>>>(
            d_centroides, d_sums, d_counts, n_lists, dim);
    }
    cudaDeviceSynchronize();

    // --- 5. Traer labels y centroides a host para construir el índice invertido ---
    std::vector<int> labels(num_vectors);
    std::vector<float> centroides(static_cast<size_t>(n_lists) * dim);
    cudaMemcpy(labels.data(), d_labels, num_vectors * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroides.data(), d_centroides, centroides.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::vector<int>> cluster_to_indices(n_lists);
    for (int i = 0; i < num_vectors; ++i) {
        cluster_to_indices[labels[i]].push_back(i);
    }

    // --- 6. n_probe centroides más cercanos a la query (n_lists es chico, se hace en host) ---
    std::vector<std::pair<float, int>> dist_query_centroids(n_lists);
    for (int c = 0; c < n_lists; ++c) {
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = query_embedding[d] - centroides[static_cast<size_t>(c) * dim + d];
            dist += diff * diff;
        }
        dist_query_centroids[c] = {dist, c};
    }
    std::partial_sort(dist_query_centroids.begin(),
                       dist_query_centroids.begin() + n_probe,
                       dist_query_centroids.end());

    // --- 7. Consolidar índices de los clusters seleccionados ---
    std::vector<int> filtered_indices;
    for (int p = 0; p < n_probe; ++p) {
        int c_id = dist_query_centroids[p].second;
        filtered_indices.insert(filtered_indices.end(),
                                 cluster_to_indices[c_id].begin(),
                                 cluster_to_indices[c_id].end());
    }
    int num_filtered = static_cast<int>(filtered_indices.size());

    if (num_filtered == 0) {
        cudaFree(d_dataset); cudaFree(d_query); cudaFree(d_centroides);
        cudaFree(d_sums); cudaFree(d_labels); cudaFree(d_counts);
        return {};
    }

    // --- 8. Distancias exactas con el kernel compute_distances ---
    int *d_filtered_indices;
    float *d_distances;
    cudaMalloc(&d_filtered_indices, num_filtered * sizeof(int));
    cudaMalloc(&d_distances, num_filtered * sizeof(float));
    cudaMemcpy(d_filtered_indices, filtered_indices.data(), num_filtered * sizeof(int), cudaMemcpyHostToDevice);

    int blocks_filtered = (num_filtered + threads - 1) / threads;
    compute_distances<<<blocks_filtered, threads>>>(
        d_query, d_dataset, d_filtered_indices, d_distances, dim, num_filtered);
    cudaDeviceSynchronize();

    std::vector<float> distances(num_filtered);
    cudaMemcpy(distances.data(), d_distances, num_filtered * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 9. Top-K por distancia ascendente ---
    std::vector<int> order(num_filtered);
    std::iota(order.begin(), order.end(), 0);
    int k = std::min(top_k, num_filtered);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                       [&](int a, int b) { return distances[a] < distances[b]; });

    std::vector<std::string> context_for_llm;
    context_for_llm.reserve(k);
    for (int i = 0; i < k; ++i) {
        int global_idx = filtered_indices[order[i]];
        context_for_llm.push_back(lista_textos[global_idx]);
    }

    // --- 10. Liberar memoria GPU ---
    cudaFree(d_dataset); cudaFree(d_query); cudaFree(d_centroides);
    cudaFree(d_sums); cudaFree(d_labels); cudaFree(d_counts);
    cudaFree(d_filtered_indices); cudaFree(d_distances);

    return context_for_llm;
}
