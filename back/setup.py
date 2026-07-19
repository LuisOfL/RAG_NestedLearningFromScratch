from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Compilar dentro de WSL/Ubuntu con:
#   python setup.py build_ext --inplace
# Luego en Python:
#   import ivf_search
#   resultado = ivf_search.buscar_ivf_gpu(lista_embeddings, query_embedding, lista_textos)

setup(
    name='ivf_search',
    ext_modules=[
        CUDAExtension('ivf_search', ['bindings.cpp', 'ivf_search.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
