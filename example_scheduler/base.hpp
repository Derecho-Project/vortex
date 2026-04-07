#include <cascade/user_defined_logic_interface.hpp>

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
