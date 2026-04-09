/** The responsibility of the TaskJoinService is to act as a temporary holding arena which emits computable
 * tasks (tasks that have all dependencies present). It contains 4 internal pieces
 *  1. Input Arena      - size-segmented arena allocator. see arena.hpp
 *  2. Ingress Queue    - TODO: jq54: writeme.
 *  3. Dag Registry     - parses `dfgs.json` into a deteterministic execution graph
 *  4. Join Table       - tracks which dependency slots have been satisfied and emits as `TaskBinding` when all
 *                        all uptream components are present
 */
#pragma once
#include <vortex_scheduler/core.hpp>
#include <vortex_scheduler/arena.hpp>
#include <vortex_scheduler/dag_registry.hpp>
#include <vortex_scheduler/join_table.hpp>
#include <atomic>
#include <deque>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

/// @brief A temporary holding arena which emits computatble tasks once all dependencies are present
class TaskJoinService {
  public:
    
    TaskJoinService(const DagRegistry &dag_registry);
    TaskJoinService() = delete;
    TaskJoinService(const TaskJoinService &task_join_service) = delete;
    TaskJoinService(const TaskJoinService &&task_join_service) = delete;
    ~TaskJoinService() = default;

  public:
    /// @brief receive a message from path `string key` and an encoded `TaskOutput`
    std::optional<TaskBinding> recv(const std::string_view& string_key, const std::span<const std::byte> &bytes);

    /// @brief non-blocking ingress enqueue used by producer threads
    bool try_ingest(const std::string_view& string_key, const std::span<const std::byte>& bytes);

    /// @brief owner-thread method that processes enqueued ingress packets
    std::size_t drain_ingress(std::size_t max_messages = 64);

    /// @brief owner-thread method that returns one ready task binding if available
    std::optional<TaskBinding> poll_ready();

    /// @brief resolve a BlobHandle emitted by recv into a payload span
    std::optional<std::span<const std::byte>> resolve(const BlobHandle& handle) const;

    /// @brief release payload storage associated with a previously emitted binding
    void free(const TaskBinding& binding);

    const DagRegistry& dag() const noexcept { return _dag_registry; }

  private:
    struct IngressPacket {
      std::string key;
      std::vector<std::byte> bytes;
    };

    struct IngressSlot {
      std::atomic<std::size_t> seq{0};
      IngressPacket packet;
    };

    bool try_enqueue_ingress(IngressPacket&& packet);
    bool try_dequeue_ingress(IngressPacket& packet);
    std::optional<TaskBinding> process_packet(const std::span<const std::byte>& bytes);
    static std::size_t next_pow2(std::size_t value);
    bool claim_owner_thread();

  private:
    DagRegistry _dag_registry;
    JoinTable   _join_table;
    ArenaAllocator<uint64_t> _payload_arena;
    std::unordered_map<uint64_t, BufferHandle> _payload_handles;
    std::deque<TaskBinding> _ready_bindings;

    std::unique_ptr<IngressSlot[]> _ingress_ring;
    std::size_t _ingress_capacity = 0;
    std::size_t _ingress_mask = 0;
    std::atomic<std::size_t> _ingress_head{0};
    std::atomic<std::size_t> _ingress_tail{0};

    std::atomic<std::size_t> _owner_thread_token{0};
    uint64_t _next_payload_id = 1;
};

VORTEX_SCHEDULER_NAMESPACE_END
