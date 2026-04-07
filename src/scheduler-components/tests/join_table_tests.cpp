#include <catch2/catch.hpp>

#include <stdexcept>

#include <vortex_scheduler/join_table.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

BlobHandle make_handle(uint32_t seg, uint32_t off, uint32_t size) {
  BlobHandle h;
  h.pool_class = 1;
  h.segment_id = seg;
  h.offset = off;
  h.size = size;
  return h;
}

}  // namespace

TEST_CASE("JoinTable emits binding only when all dependencies arrive", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 7, 42};

  auto first = table.add_input(task, 3, 2, 0, make_handle(1, 0, 8));
  CHECK_FALSE(first.has_value());
  CHECK(table.pending_tasks() == 1);

  auto second = table.add_input(task, 3, 2, 1, make_handle(1, 8, 8));
  REQUIRE(second.has_value());
  CHECK(second->task.job_id == 7);
  CHECK(second->task.task_id == 42);
  CHECK(second->processor_id == 3);
  REQUIRE(second->inputs.size() == 2);
  CHECK(second->inputs[0].offset == 0);
  CHECK(second->inputs[1].offset == 8);
  CHECK(table.pending_tasks() == 0);
}

TEST_CASE("JoinTable ignores duplicate dependency slot", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 8, 11};

  auto first = table.add_input(task, 5, 2, 0, make_handle(2, 0, 4));
  CHECK_FALSE(first.has_value());

  auto dup = table.add_input(task, 5, 2, 0, make_handle(2, 4, 4));
  CHECK_FALSE(dup.has_value());
  CHECK(table.pending_tasks() == 1);

  auto second = table.add_input(task, 5, 2, 1, make_handle(2, 8, 4));
  REQUIRE(second.has_value());
  REQUIRE(second->inputs.size() == 2);
  CHECK(second->inputs[0].offset == 0);
  CHECK(second->inputs[1].offset == 8);
}

TEST_CASE("JoinTable rejects invalid slot and expected_inputs", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 1, 1};

  REQUIRE_THROWS_AS(
      table.add_input(task, 1, 0, 0, make_handle(1, 0, 1)),
      std::invalid_argument);

  REQUIRE_THROWS_AS(
      table.add_input(task, 1, 2, 2, make_handle(1, 0, 1)),
      std::out_of_range);
}

TEST_CASE("JoinTable enforces processor and expected count consistency", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 5, 9};

  auto first = table.add_input(task, 2, 3, 0, make_handle(3, 0, 2));
  CHECK_FALSE(first.has_value());

  REQUIRE_THROWS_AS(
      table.add_input(task, 99, 3, 1, make_handle(3, 2, 2)),
      std::invalid_argument);

  REQUIRE_THROWS_AS(
      table.add_input(task, 2, 2, 1, make_handle(3, 2, 2)),
      std::invalid_argument);
}

TEST_CASE("JoinTable can be reused for same task after emission", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 4, 4};

  auto b1 = table.add_input(task, 6, 1, 0, make_handle(10, 0, 16));
  REQUIRE(b1.has_value());
  CHECK(table.pending_tasks() == 0);

  auto b2 = table.add_input(task, 6, 1, 0, make_handle(11, 16, 16));
  REQUIRE(b2.has_value());
  REQUIRE(b2->inputs.size() == 1);
  CHECK(b2->inputs[0].segment_id == 11);
}

TEST_CASE("JoinTable clear_task drops partial state", "[join_table]") {
  JoinTable table;
  TaskRef task{0, 2, 3};

  auto first = table.add_input(task, 1, 2, 0, make_handle(4, 0, 1));
  CHECK_FALSE(first.has_value());
  CHECK(table.pending_tasks() == 1);

  table.clear_task(task);
  CHECK(table.pending_tasks() == 0);
}
