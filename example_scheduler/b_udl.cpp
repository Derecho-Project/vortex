#include <vortex_scheduler/udl_base.hpp>

#include "scheduler_messages.hpp"

using namespace scheduler;

namespace derecho {
namespace cascade {


#define MY_UUID "a945d81b-53c7-43c3-b461-12c85d6883ab"
#define MY_DESC "Demo DLL UDL: tag the upstream StepA message with a [B] prefix and forward."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

class TaskB_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override { }

	void execute_udl(TaskBinding && binding) override {
		// Decode the single upstream StepA message (zero-copy view into arena).
		auto upstream_view = StepAMessageView::from_buffer(view(binding.payload_handles[0]));

		StepBMessage out{
			.metadata = upstream_view.metadata,
			.message  = std::string("[B] ") + std::string(upstream_view.message),
		};

		notify_finish(binding);
		emit<StepBMessage>(std::move(binding), std::move(out));
	}

public:
	TaskB_OCDPO() : VortexWorkerUdl("/B", "/B/SCHED", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskB_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskB_OCDPO)
} // namespace cascade
} // namespace derecho

