// bindings.cpp
// Expone buscar_ivf_gpu (definida en ivf_search.cu) a Python vía pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

// Declarada en ivf_search.cu
std::vector<std::string> buscar_ivf_gpu(
    const std::vector<std::vector<float>>& lista_embeddings,
    const std::vector<float>& query_embedding,
    const std::vector<std::string>& lista_textos,
    int n_lists,
    int n_probe,
    int top_k,
    int kmeans_iters
);

namespace py = pybind11;

PYBIND11_MODULE(ivf_search, m) {
    m.def(
        "buscar_ivf_gpu",
        &buscar_ivf_gpu,
        py::arg("lista_embeddings"),
        py::arg("query_embedding"),
        py::arg("lista_textos"),
        py::arg("n_lists") = 4,
        py::arg("n_probe") = 2,
        py::arg("top_k") = 5,
        py::arg("kmeans_iters") = 10,
        "Busqueda IVF sobre embeddings usando CUDA C++"
    );
}
