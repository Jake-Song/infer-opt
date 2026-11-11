"""Add isolated Qwen3.5 MoE cases and measurement hooks to a pinned llama.cpp."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = 'a0ad695cb212db3850f163121ddc687513448d42'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    args = p.parse_args()
    root = args.source
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    test = root / 'tests/test-backend-ops.cpp'
    code = test.read_text()
    assert 'infer_moe_cases' not in code
    code = '#include <cuda_runtime_api.h>\n#include "ggml-cuda.h"\n' + code
    inc = Path(__file__).with_name('bench.inc').read_text()
    marker = 'static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {'
    assert marker in code
    code = code.replace(marker, inc + '\n' + marker, 1)
    for name in ['eval', 'perf']:
        marker = f'static std::vector<std::unique_ptr<test_case>> make_test_cases_{name}() {{'
        code = code.replace(marker, marker + '\n    if (std::getenv("INFER_MOE_CASES")) return infer_moe_cases();', 1)
    marker = 'int main(int argc, char ** argv) {'
    code = code.replace(marker, marker + '\n    if (std::getenv("INFER_MOE_MEASURE")) return infer_moe_measure();', 1)
    test.write_text(code)
    cuda = root / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    cuda_code = cuda.read_text()
    marker = '    CUDA_CHECK(cudaGraphDestroy(cuda_graph));'
    assert cuda_code.count(marker) == 1
    cuda_code = cuda_code.replace(marker, '''
    if (const char * directory = getenv("INFER_MOE_DOT_DIR")) {
        const char * name = getenv("INFER_MOE_CASE_NAME");
        char filename[4096];
        snprintf(filename, sizeof(filename), "%s/%s-%zu.dot", directory, name ? name : "capture", idx);
        CUDA_CHECK(cudaGraphDebugDotPrint(cuda_graph, filename, cudaGraphDebugDotFlagsVerbose));
    }
''' + marker)
    cuda.write_text(cuda_code + '''
// Measurement-only accessors. Kernel implementations and dispatch are unchanged.
extern "C" GGML_BACKEND_API void * infer_moe_cuda_stream(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return (void *) ctx->stream();
}
extern "C" GGML_BACKEND_API void * infer_moe_cuda_graph(ggml_backend_t backend, ggml_cgraph * gf) {
#ifdef USE_CUDA_GRAPH
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    auto * graph = ctx->cuda_graph(ggml_cuda_graph_get_key(gf));
    if (graph->instances.size() != 1) return nullptr;
    return (void *) graph->instances[0];
#else
    return nullptr;
#endif
}
__global__ static void infer_moe_prelude_kernel() {
    const unsigned long long start = clock64();
    while (clock64() - start < 300000ULL) {}
}
extern "C" GGML_BACKEND_API void infer_moe_prelude(void * stream) {
    infer_moe_prelude_kernel<<<1,32,0,(cudaStream_t)stream>>>();
}
''')
    cmake = root / 'tests/CMakeLists.txt'
    cmake.write_text(cmake.read_text() + '\nfind_package(CUDAToolkit REQUIRED)\ntarget_link_libraries(test-backend-ops PRIVATE CUDA::cudart)\n')
    print(json.dumps({'revision': REVISION, 'test_sha256': hashlib.sha256(test.read_bytes()).hexdigest(),
                      'cuda_sha256': hashlib.sha256(cuda.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
