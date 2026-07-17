import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

cuda_home = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6')
nvcc_bin = os.path.join(cuda_home, 'bin', 'nvcc.exe')

extra_includes = []
extra_libs = []
extra_link = []

ext_modules = [
    Extension(
        'cpp_backend',
        sources=['utils_backend.cpp'],  
        include_dirs=[
            pybind11.get_include(),
            os.path.join(cuda_home, 'include')
        ] + extra_includes,
        library_dirs=[
            os.path.join(cuda_home, 'lib', 'x64')
        ] + extra_libs,
        libraries=['cudart'] + extra_link,
        language='c++',
    ),
]

class CUDA_Compatible_BuildExt(build_ext):
    def build_extensions(self):
        # Guardamos el compilador original de C++ de Visual Studio (cl.exe)
        original_compile = self.compiler._compile
        
        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Leemos las primeras líneas del archivo para saber si contiene código CUDA real
            is_cuda = False
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '__global__' in content or '<<<' in content:
                        is_cuda = True
            except Exception:
                pass

            if is_cuda:
                # Si contiene sintaxis CUDA, forzamos la compilación con NVCC
                includes = [f'-I{d}' for d in self.compiler.include_dirs]
                
                # Argumentos de optimización para NVCC en Windows
                nvcc_args = [
                    '-c', src,
                    '-o', obj,
                    '-O3',
                    '-std=c++17',
                    '--compiler-options', '/MD,/EHsc'
                ] + includes
                
                print(f"\n[CUDA Compiling] {src} usando NVCC...")
                self.compiler.spawn([nvcc_bin] + nvcc_args)
            else:
                # Si es un archivo C++ puro (sin código CUDA), usamos Visual Studio
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
                
        # Inyectamos nuestra función de compilación inteligente
        self.compiler._compile = custom_compile
        
        # Flags globales para el enlazador de Windows (MSVC)
        for ext in self.extensions:
            ext.extra_compile_args = ['/O2', '/std:c++17', '/EHsc']
            
        super().build_extensions()

setup(
    name='cpp_backend',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDA_Compatible_BuildExt},
    zip_safe=False,
)