#include <vortex_scheduler/messages.hpp>

namespace scheduler::message {

// Explicit template instantiations so the symbols live in libvortex_scheduler.so
// rather than being emitted in every translation unit that includes messages.hpp.
template struct SchedulerCommandT<serde::ViewFamily>;
template struct SchedulerCommandT<serde::OwnedFamily>;

template struct WorkerStatusT<serde::ViewFamily>;
template struct WorkerStatusT<serde::OwnedFamily>;

template struct TaskOutputT<serde::ViewFamily>;
template struct TaskOutputT<serde::OwnedFamily>;

} // namespace scheduler::message
