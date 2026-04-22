#include <vortex_scheduler/udl_base.hpp>

#include "scheduler_messages.hpp"

using namespace scheduler;

namespace derecho {
namespace cascade {


#define MY_UUID "4f64e67a-44dd-4721-967d-d6039cc32470"
#define MY_DESC "Demo DLL UDL: tag the upstream StepA message with a [C] prefix and forward."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

class TaskC_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override { }

	void execute_udl(TaskBinding && binding) override {
		auto upstream_view = StepAMessageView::from_buffer(view(binding.payload_handles[0]));

		StepCMessage out{
			.metadata = upstream_view.metadata,
			.message  = std::string("[C] ") + std::string(upstream_view.message),
		};

		notify_finish(binding);
		emit<StepCMessage>(std::move(binding), std::move(out));
	}

public:
	TaskC_OCDPO() : VortexWorkerUdl("/C", "/C/SCHED", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskC_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskC_OCDPO)
} // namespace cascade
} // namespace derecho
