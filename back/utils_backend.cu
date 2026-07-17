#include <poppler-document.h>
#include <poppler-page.h>
#include <string>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

namespace py = pybind11;

// --- EXTRACCIÓN DE TEXTO (Añadida de vuelta) ---
std::string extract_text_from_pdf(const std::string& pdf_path) {
    poppler::document* doc = poppler::document::load_from_file(pdf_path);
    if (!doc) {
        throw std::runtime_error("Error al abrir el PDF");
    }
    
    std::string full_text = "";
    int num_pages = doc->pages();
    for (int i = 0; i < num_pages; ++i) {
        poppler::page* p = doc->create_page(i);
        if (p) {
            full_text += p->text().to_utf8().data();
            delete p;
        }
    }
    delete doc;
    return full_text;
}

// --- TEXT SPLITTER ---
std::vector<std::string> recursive_character_text_splitter(const std::string& text, size_t chunk_size, size_t chunk_overlap) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < text.length()) {
        size_t end = std::min(start + chunk_size, text.length());
        chunks.push_back(text.substr(start, end - start));
        start += chunk_size - chunk_overlap;
    }
    return chunks;
}

// --- CURL WRITE CALLBACK ---
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// --- OPENAI API EMBEDDINGS ---
std::vector<float> get_embedding_from_api(const std::string& text, const std::string& api_key) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    std::vector<float> embedding;

    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/embeddings");
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string auth = "Authorization: Bearer " + api_key;
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        nlohmann::json j;
        j["input"] = text;
        j["model"] = "text-embedding-3-small";
        std::string data = j.dump();

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        auto response_json = nlohmann::json::parse(readBuffer);
        embedding = response_json["data"][0]["embedding"].get<std::vector<float>>();
    }
    return embedding;
}

// --- CUDA KERNEL ---
__global__ void assign_clusters_kernel(const float* d_embeddings, const float* d_centroids, int* d_labels, int num_embeddings, int dim, int k) {
    int idx = blockIdx::x * blockDim::x + threadIdx::x;
    if (idx >= num_embeddings) return;

    int best_cluster = 0;
    float min_dist = 1e10f; 

    for (int cluster = 0; cluster < k; ++cluster) {
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = d_embeddings[idx * dim + d] - d_centroids[cluster * dim + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = cluster;
        }
    }
    d_labels[idx] = best_cluster;
}

// --- K-MEANS CUDA ENTRYPOINT ---
std::vector<int> kmeans_cuda(const std::vector<float>& h_embeddings, int num_embeddings, int dim, int k) {
    std::vector<int> h_labels(num_embeddings);
    float *d_embeddings, *d_centroids;
    int *d_labels;

    std::vector<float> h_centroids(k * dim);
    for(int i = 0; i < k * dim; ++i) h_centroids[i] = h_embeddings[i]; 
    
    cudaMalloc(&d_embeddings, num_embeddings * dim * sizeof(float));
    cudaMalloc(&d_centroids, k * dim * sizeof(float));
    cudaMalloc(&d_labels, num_embeddings * sizeof(int));

    cudaMemcpy(d_embeddings, h_embeddings.data(), num_embeddings * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), k * dim * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_embeddings + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < 10; ++iter) {
        assign_clusters_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_embeddings, d_centroids, d_labels, num_embeddings, dim, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_labels.data(), d_labels, num_embeddings * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_embeddings);
    cudaFree(d_centroids);
    cudaFree(d_labels);

    return h_labels;
}

// ── REGISTRO DE EXPORTACIÓN PARA PYBIND11 ─────────────────────────────────
PYBIND11_MODULE(cpp_backend, m) {
    m.doc() = "Backend de procesamiento de documentos y clustering CUDA";
    
    m.def("extract_text_from_pdf", &extract_text_from_pdf, "Extrae texto de un archivo PDF vía Poppler");
    m.def("recursive_character_text_splitter", &recursive_character_text_splitter, "Divide texto en Chunks");
    m.def("get_embedding_from_api", &get_embedding_from_api, "Obtiene embeddings usando libcurl");
    m.def("kmeans_cuda", &kmeans_cuda, "Ejecuta K-means en la GPU mediante CUDA");
}