#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <memory>
#include <pyscheduler/pyscheduler.hpp>
#include <string>
#include <unordered_map>

namespace derecho {
namespace cascade {

#define MY_UUID "48e60f7c-8500-11eb-8755-0242ac110002"
#define MY_DESC "Demo DLL UDL that prints received messages and invokes Python print via PyScheduler."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

namespace {
std::unique_ptr<pyscheduler::PyManager> g_python_manager;
std::unique_ptr<pyscheduler::PyManager::InvokeHandler> g_print_handler;
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

        std::cout << "[EXAMPLE printer ocdpo]: I(" << worker_id
                  << ") received an object from sender:" << sender
                  << " with key=" << key_string
                  << ", matching prefix=" << key_string.substr(0, prefix_length)
                  << std::endl;

        if (g_print_handler) {
            g_print_handler->invoke<void>(
                std::string("hello world from pyscheduler, key=") + key_string);
        }
    }

    static void initialize() {
        if (!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ExampleOCDPO>();
        }

        if (!g_python_manager) {
            g_python_manager = std::make_unique<pyscheduler::PyManager>();
            g_python_manager->add_path("python_udls");
        }

        if (!g_print_handler) {
            auto handler = g_python_manager->loadPythonModule("hello_world", "invoke");
            g_print_handler =
                std::make_unique<pyscheduler::PyManager::InvokeHandler>(std::move(handler));
            g_print_handler->invoke<void>("initialized");
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