#include <memory>
#include <pybind11/pybind11.h>
#include <pyscheduler/pyscheduler.hpp>
#include <string_view>
#include <vortex_scheduler/udl_base.hpp>

#include "scheduler_messages.hpp"

using namespace scheduler;

namespace derecho {
namespace cascade {

#define MY_UUID "24e10f1c-1100-11eb-1111-0111ac110002"
#define MY_DESC "Demo DLL UDL: on each message, build 256x256 GPU tensors and call Python add_gpu."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

struct global_state {
	pyscheduler::PyManager::InvokeHandler step_a;
};

std::unique_ptr<global_state> globals = nullptr;

class TaskA_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override {
		pyscheduler::PyManager manager;
		manager.add_path(".");

		globals = std::make_unique<global_state>(
			global_state{ .step_a = manager.loadPythonModule("python_udls.step_a", "step_a", 16, 1) });
	}

	pybind11::object commit_fn(const MetadataA& metadata, const TaskBinding& binding) {
		// Parse the serialized StepAMessage and forward only its string payload.
		notify_finish(binding);

		auto payload = view(binding.payload_handles[0]);
		auto message = StepAMessageView::from_buffer(payload);
		return pybind11::str(std::string(message.message));
	}

	void callback_fn(const MetadataA& metadata, TaskBinding&& binding, pybind11::object&& result) {
		// object is converted into C++ representation and moved into the emit call which serializes it into the output buffer and sends it to the next task.
		std::string result_str = result.cast<std::string>();
		emit<StepAMessage>(std::move(binding), std::move(StepAMessage{
			.metadata = metadata,
			.message = result_str,
		}));
	}

	void execute_udl(TaskBinding&& binding) override {
		// example of how you would bind the metadata to a closure
		auto payload = view(binding.payload_handles[0]);
		auto metadata = StepAMessageView::from_buffer(payload).metadata;

		auto commit_closure = [this, metadata, binding=binding]() {
			return commit_fn(metadata, binding);
		};

		auto callback_closure = [this, metadata, binding=binding](pybind11::object&& result) mutable {
			return callback_fn(metadata, std::move(binding), std::move(result));
		};

		auto future = globals->step_a.queue_invoke(commit_closure, callback_closure);
	}

public:
	TaskA_OCDPO()
		: VortexWorkerUdl("/A", "/A/SCHED", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskA_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskA_OCDPO)
} // namespace cascade
} // namespace derecho
