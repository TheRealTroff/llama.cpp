set(TARGET llama-gguf-repack)
add_executable(${TARGET} gguf-repack.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# ggml-quants.h: the SoA row packers shared with the CPU reference paths
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
