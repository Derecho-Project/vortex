#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include <cascade/object.hpp>

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/pytypes.h>
#include <pyscheduler/pyscheduler.hpp>
#include <pyscheduler/tensor.hpp>
#include <spdlog/logger.h>

#include <cascade/user_defined_logic_interface.hpp>
#include <vortex_scheduler/prelude.hpp>

// Generates the standard UDL DLL entrypoints for an OCDPO type.
// The OCDPO type must provide static initialize() and get() methods,
// and declare `static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;`.
#define VORTEX_DEFINE_UDL_ENTRYPOINTS(OCDPOType)                                                   \
	std::shared_ptr<OffCriticalDataPathObserver> OCDPOType::ocdpo_ptr;                             \
	void initialize(ICascadeContext* ctxt) {                                                       \
		(void)ctxt;                                                                                \
		OCDPOType::initialize();                                                                   \
	}                                                                                              \
	std::shared_ptr<OffCriticalDataPathObserver> get_observer(ICascadeContext* ctxt,               \
															  const nlohmann::json& cfg) {         \
		(void)ctxt;                                                                                \
		(void)cfg;                                                                                 \
		return OCDPOType::get();                                                                   \
	}                                                                                              \
	void release(ICascadeContext* ctxt) {                                                          \
		(void)ctxt;                                                                                \
	}

// Generates UUID/description exports from MY_UUID and MY_DESC.
// Define both MY_UUID and MY_DESC in the UDL .cpp before invoking this macro.
#define VORTEX_DEFINE_UDL_METADATA(uuid, desc)                                                     \
	std::string get_uuid() {                                                                       \
		return uuid;                                                                               \
	}                                                                                              \
	std::string get_description() {                                                                \
		return desc;                                                                               \
	}

#define VORTEX_DEFINE_CLASS_METHODS(OCDPOType)                                                     \
private:                                                                                           \
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;                                 \
                                                                                                   \
public:                                                                                            \
	static void initialize() {                                                                     \
		if(!ocdpo_ptr) {                                                                           \
			ocdpo_ptr = std::make_shared<OCDPOType>();                                             \
		}                                                                                          \
	}                                                                                              \
                                                                                                   \
	static std::shared_ptr<OffCriticalDataPathObserver> get() {                                    \
		return ocdpo_ptr;                                                                          \
	}

namespace derecho {
namespace cascade {

class VortexWorkerUdl : public OffCriticalDataPathObserver {
public:
	VortexWorkerUdl() = delete;
	VortexWorkerUdl(const std::string_view& name,
					const std::string_view& data_path,
					const std::string_view& scheduler_path,
					const std::string_view& uuid)
		: _name(name)
		, _data_path(data_path)
		, _scheduler_path(scheduler_path)
		, _uuid(uuid) { }

private:
	/// @brief name of this udl
	std::string _name;

	/// @brief path in which this udl receives data messages
	std::string _data_path;

	/// @brief path in which this udl receives scheduling messages
	std::string _scheduler_path;

	/// @brief uuid of of this udl that is not locked behind a compile time macro
	std::string _uuid;

	/// @brief initialization guard
	bool _initialized = false;

protected:
	std::unique_ptr<scheduler::DagRegistry> _registry;
	std::unique_ptr<scheduler::TaskJoinService> _join_service;

protected:
	// methods which the child class should overload

	/// @brief called once the first time UDL receives data. purposes is to lazily load resource intensive computation units.
	virtual void initialize_resources() = 0;

	/// @brief dispatched for a task when upstream dependencies are satisfied
	virtual void execute_udl(scheduler::TaskBinding binding) = 0;

protected:
	void finish(scheduler::TaskBinding ingress_binding, );

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

		if(!_initialized) {
			_initialized = true;
			_registry = std::make_unique<scheduler::DagRegistry>(
				scheduler::DagRegistry::from_dfg_file("jobs.json", _uuid));
			_join_service = std::make_unique<scheduler::TaskJoinService>(*_registry);
			initialize_resources();
			spdlog::info("[Worker:{}]: initialized", _name);
		}

		const auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
		if(!typed_ctxt) {
			spdlog::error("[Worker:{}]: empty cascade context", _name);
			return;
		}

		const auto obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
		if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
			spdlog::error("[Worker:{}]: empty blob object", _name);
			return;
		}

		// at this point, the blob object either encodes a task output or scheduler command
		// the way to differentiate between to two is to compare path prefixes (_data_path) vs.
		// _scheduler_path

		// NOTE: std::span is a slice type, meaning it is a non-owning, non-writable view into a slice of data.
		// Since the view is a slice within the SST table, we must extend the lifetime via a memcpy if we want
		// to process the data outside the lifetime of this method.
		const std::string_view path_prefix(
			key_string.data(),
			prefix_length <= key_string.size() ? prefix_length : key_string.size());
		const std::span<const std::byte> payload_slice(
			reinterpret_cast<const std::byte*>(obj->blob.bytes), obj->blob.size);
		if(path_prefix == _data_path) {
			auto binding = _join_service->recv(key_string, payload_slice);
			if(binding) {
				// safe use of "unsafe" method because we check if it exists in the guard
				execute_udl(*binding);
				_join_service->free(*binding);
			}
		} else if(path_prefix == _scheduler_path) {
		} else {
			spdlog::error("[Worker:{}]: invalid path prefix: {}", _name, path_prefix);
			return;
		}
	}
};

} // namespace cascade
} // namespace derecho
