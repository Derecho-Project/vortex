#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <memory>
#include <pyscheduler/pyscheduler.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <cstdint>

#include <pyscheduler/tensor.hpp>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/pytypes.h>

namespace derecho {
namespace cascade {

#define MY_UUID "24e10f1c-1100-11eb-1111-0111ac110002"
#define MY_DESC "Demo DLL UDL: on each message, build 256x256 GPU tensors and call Python add_gpu."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

namespace {
std::unique_ptr<pyscheduler::PyManager> g_python_manager;

std::unique_ptr<pyscheduler::PyManager::InvokeHandler> g_add_gpu_handler;

constexpr int kDim = 256;

std::vector<float> generateRandomMatrix(int rows, int cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(static_cast<size_t>(rows * cols));
    for (float& v : out) {
        v = dist(rng);
    }
    return out;
}

pybind11::capsule toDlpackCapsule(DLManagedTensor* managed_tensor) {
    return pybind11::capsule(managed_tensor, "dltensor", [](PyObject* capsule_ptr) {
        auto* tensor =
            reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_ptr, "dltensor"));
        if (tensor != nullptr && tensor->deleter != nullptr) {
            tensor->deleter(tensor);
        }
    });
}
}  // namespace

class ExampleOCDPO : public OffCriticalDataPathObserver {
public:
    void operator()(const derecho::node_id_t sender,
                    const std::string& key_string,
                    const uint32_t prefix_length,
                    persistent::version_t version,
                    const mutils::ByteRepresentable* const value_ptr,
                    const std::unordered_map<std::string, bool>& outputs,
                    ICascadeContext* ctxt,
                    uint32_t worker_id) override {
        (void)version;
        (void)value_ptr;
        (void)outputs;
        (void)ctxt;

        std::cout << "[EXAMPLE ocdpo]: I(" << worker_id
                  << ") received object from sender=" << sender
                  << " key=" << key_string
                  << " prefix=" << key_string.substr(0, prefix_length)
                  << std::endl;

        if (!g_add_gpu_handler) {
            std::cerr << "[EXAMPLE ocdpo]: Python add_gpu handler is not initialized." << std::endl;
            return;
        }

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cerr << "[EXAMPLE ocdpo]: No CUDA device available." << std::endl;
            return;
        }

        const uint32_t seed_a = static_cast<uint32_t>(sender) ^ worker_id ^ 0x1234u;
        const uint32_t seed_b = seed_a ^ 0x9e3779b9u;

        auto a_host = generateRandomMatrix(kDim, kDim, seed_a);
        auto b_host = generateRandomMatrix(kDim, kDim, seed_b);

        try {
            pybind11::gil_scoped_acquire gil;
            auto a_capsule = toDlpackCapsule(pyscheduler::createCudaMatrixDlpack(a_host, kDim, kDim));
            auto b_capsule = toDlpackCapsule(pyscheduler::createCudaMatrixDlpack(b_host, kDim, kDim));

            auto parse_result = [](pybind11::object&& obj) {
                return obj.cast<double>();
            };

            // Python computes (A + B).sum() and returns a scalar.
            double sum = g_add_gpu_handler->invoke(parse_result, a_capsule, b_capsule);

            std::cout << "[EXAMPLE ocdpo]: GPU add computed; sum=" << sum << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "[EXAMPLE ocdpo]: GPU compute failed: " << ex.what() << std::endl;
        }
    }

    static void initialize() {
        if (!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ExampleOCDPO>();
        }

        if (!g_python_manager) {
            g_python_manager = std::make_unique<pyscheduler::PyManager>();
            g_python_manager->add_path("python_udls");
            g_python_manager->add_path("/home/yy354/.local/lib/python3.10/site-packages");
            // g_python_manager->add_path("/usr/lib/python3.10/lib-dynload");
            // g_python_manager->add_path("/home/yy354/workspace/FLMR");
            // g_python_manager->add_path("/home/yy354/workspace/vortex/build-Debug");
            // g_python_manager->add_path("/usr/lib/python310.zip");
            // g_python_manager->add_path("/usr/lib/python3.10");
            // g_python_manager->add_path("/usr/lib/python3.10/lib-dynload");
            // g_python_manager->add_path("/usr/local/lib/python3.10/dist-packages");
            // g_python_manager->add_path("/usr/lib/python3/dist-packages");
        }

        if (!g_add_gpu_handler) {
            auto handler = g_python_manager->loadPythonModule("add_gpu", "invoke");
            g_add_gpu_handler =
                std::make_unique<pyscheduler::PyManager::InvokeHandler>(std::move(handler));
        }
    }

    static std::shared_ptr<OffCriticalDataPathObserver> get() {
        return ocdpo_ptr;
    }

private:
    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
};

std::shared_ptr<OffCriticalDataPathObserver> ExampleOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    (void)ctxt;
    ExampleOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
    ICascadeContext* ctxt, const nlohmann::json& cfg) {
    (void)ctxt;
    (void)cfg;
    return ExampleOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    (void)ctxt;
}

}  // namespace cascade
}  // namespace derecho