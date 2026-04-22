#include <iostream>

#include <vortex_scheduler/udl_base.hpp>

#include "scheduler_messages.hpp"

using namespace scheduler;

namespace derecho {
namespace cascade {


#define MY_UUID "2bc87cc7-7ade-43d1-9a47-2a497b2fd5c0"
#define MY_DESC "Demo DLL UDL: join upstream B and C messages and print the combined result."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

class TaskD_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override { }

	void execute_udl(TaskBinding && binding) override {
		// D is a join of B and C; the join thread guarantees both payloads
		// are present in payload_handles before dispatch. The B/C upstream
		// types are wire-compatible with StepAMessage, so we decode either
		// payload as StepBMessageView/StepCMessageView only as needed; here
		// we just read the message string.
		std::string combined;
		combined.reserve(128);
		for(std::uint8_t i = 0; i < binding.num_upstream; ++i) {
			auto upstream_view = StepBMessageView::from_buffer(view(binding.payload_handles[i]));
			if(i > 0) combined.append(" + ");
			combined.append(std::string(upstream_view.message));
		}

		std::cout << "[D] graph=" << binding.current_task.graph_id
				  << " job=" << binding.current_task.job_id
				  << " stream=" << binding.current_task.stream_id
				  << " result=" << combined << std::endl;

		notify_finish(binding);
	}

public:
	TaskD_OCDPO() : VortexWorkerUdl("/D", "/D/SCHED", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskD_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskD_OCDPO)
} // namespace cascade
} // namespace derecho
