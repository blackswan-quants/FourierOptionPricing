cmake_minimum_required(VERSION 3.15)
project(cpp_pricer LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_pricer fft_pricer.cpp)
target_compile_features(cpp_pricer PRIVATE cxx_std_14)

if(MSVC)
    target_compile_options(cpp_pricer PRIVATE /O2 /openmp)
else()
    target_compile_options(cpp_pricer PRIVATE -O3 -fopenmp)
    target_link_options(cpp_pricer PRIVATE -fopenmp)
endif()

# MinGW (e.g. Strawberry Perl, MSYS2 on PATH): Python cannot find its runtime DLLs
# (libstdc++, libgcc_s, libwinpthread, libgomp) and the import fails with
# "DLL load failed while importing cpp_pricer". Link them into the module instead.
if(MINGW)
    target_link_options(cpp_pricer PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Install into the root of site-packages so `import cpp_pricer` works everywhere
install(TARGETS cpp_pricer DESTINATION .)
