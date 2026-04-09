#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>

#include "messages.hpp"

VORTEX_SCHEDULER_NAMESPACE_BEGIN

struct DecisionGateKey {
	TaskRef task;
	uint16_t target_task_id = 0;

	bool operator==(const DecisionGateKey& rhs) const noexcept {
		return task.graph_id == rhs.task.graph_id &&
			   task.job_id == rhs.task.job_id &&
			   task.task_id == rhs.task.task_id &&
			   target_task_id == rhs.target_task_id;
	}
};

struct TaskRefHash {
	std::size_t operator()(const TaskRef& task) const noexcept {
		return (static_cast<std::size_t>(task.graph_id) << 32) ^
			   (static_cast<std::size_t>(task.job_id) << 16) ^
			   static_cast<std::size_t>(task.task_id);
	}
};

struct DecisionGateKeyHash {
	std::size_t operator()(const DecisionGateKey& key) const noexcept {
		const std::size_t task_hash = TaskRefHash { }(key.task);
		return (task_hash << 8) ^ static_cast<std::size_t>(key.target_task_id);
	}
};

template <typename PacketT>
class DecisionGate {
public:
	void add_credit(const DecisionGateKey& key, std::size_t amount = 1) {
		if(amount == 0) {
			return;
		}
		_credits[key] += amount;
	}

	void enqueue(const DecisionGateKey& key, PacketT&& packet) {
		_pending[key].push_back(std::move(packet));
	}

	void cancel_task(const TaskRef& task) {
		for(auto it = _pending.begin(); it != _pending.end();) {
			if(it->first.task.graph_id == task.graph_id &&
			   it->first.task.job_id == task.job_id &&
			   it->first.task.task_id == task.task_id) {
				it = _pending.erase(it);
			} else {
				++it;
			}
		}

		for(auto it = _credits.begin(); it != _credits.end();) {
			if(it->first.task.graph_id == task.graph_id &&
			   it->first.task.job_id == task.job_id &&
			   it->first.task.task_id == task.task_id) {
				it = _credits.erase(it);
			} else {
				++it;
			}
		}
	}

	template <typename SenderFn>
	std::size_t flush_key(const DecisionGateKey& key, SenderFn&& sender) {
		auto pending_it = _pending.find(key);
		if(pending_it == _pending.end()) {
			return 0;
		}

		auto credit_it = _credits.find(key);
		if(credit_it == _credits.end() || credit_it->second == 0) {
			if(pending_it->second.empty()) {
				_pending.erase(pending_it);
			}
			return 0;
		}

		auto& queue = pending_it->second;
		auto& credits = credit_it->second;
		std::size_t flushed = 0;
		while(credits > 0 && !queue.empty()) {
			auto packet = std::move(queue.front());
			queue.pop_front();
			sender(std::move(packet));
			--credits;
			++flushed;
		}

		if(queue.empty()) {
			_pending.erase(pending_it);
		}
		if(credits == 0) {
			_credits.erase(credit_it);
		}

		return flushed;
	}

	template <typename SenderFn>
	std::size_t flush(SenderFn&& sender) {
		std::size_t total = 0;
		for(auto it = _pending.begin(); it != _pending.end();) {
			const auto key = it->first;
			++it;
			total += flush_key(key, sender);
		}
		return total;
	}

	std::size_t pending_packet_count() const {
		std::size_t total = 0;
		for(const auto& [_, queue] : _pending) {
			total += queue.size();
		}
		return total;
	}

private:
	std::unordered_map<DecisionGateKey, std::deque<PacketT>, DecisionGateKeyHash> _pending;
	std::unordered_map<DecisionGateKey, std::size_t, DecisionGateKeyHash> _credits;
};

VORTEX_SCHEDULER_NAMESPACE_END
