#include "dxmt_dynamic.hpp"
#include "dxmt_texture.hpp"
#include "util_env.hpp"
#include <cstdlib>

namespace dxmt {

namespace {

/*
 * XoM low-memory patch: trim spare copies of dynamic resources.
 *
 * Stock DXMT keeps every copy made by Map(DISCARD) renaming in a per-resource
 * FIFO forever, so a resource keeps its peak copy count. The FIFO is used
 * round-robin, so the age of the front entry (how many sequence ids ago it
 * became free) grows with the number of surplus copies. When the front entry
 * has been free for longer than DXMT_DYNAMIC_SPARE_AGE sequence ids, it is a
 * surplus copy and is released. Only GPU-finished entries are touched.
 *
 * DXMT_DYNAMIC_SPARE_AGE: default 8; 0 turns the trim off (stock behaviour).
 */
uint64_t
spareAgeLimit() {
  static const uint64_t value = [] {
    auto s = env::getEnvVar("DXMT_DYNAMIC_SPARE_AGE");
    if (s.empty())
      return uint64_t(8);
    long long v = std::atoll(s.c_str());
    return v > 0 ? uint64_t(v) : uint64_t(0);
  }();
  return value;
}

constexpr unsigned kMaxTrimPerCall = 4;

template <typename Fifo>
void
trimSpare(Fifo &fifo, uint64_t coherent_seq_id) {
  uint64_t limit = spareAgeLimit();
  if (!limit)
    return;
  for (unsigned i = 0; i < kMaxTrimPerCall && !fifo.empty(); i++) {
    auto &front = fifo.front();
    if (front.will_free_at > coherent_seq_id || coherent_seq_id - front.will_free_at <= limit)
      break;
    fifo.pop_front(); // releases the allocation
  }
}

} // namespace

DynamicBuffer::DynamicBuffer(Buffer *buffer, Flags<BufferAllocationFlag> flags) :
    buffer(buffer),
    flags_(flags),
    name_(buffer->current()) {}

void
DynamicBuffer::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
DynamicBuffer::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

Rc<BufferAllocation>
DynamicBuffer::allocate(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  Rc<BufferAllocation> ret;
  for (;;) {
    if (fifo.empty()) {
      break;
    }
    auto entry = fifo.front();
    if (entry.will_free_at > coherent_seq_id) {
      break;
    }
    ret = std::move(entry.allocation);
    fifo.pop_front();
    trimSpare(fifo, coherent_seq_id);
    break;
  }
  if (!ret.ptr())
    ret = buffer->allocate(flags_);
  return ret;
}

void
DynamicBuffer::updateImmediateName(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation, uint32_t suballocation, bool owned_by_command_list) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (!owned_by_command_list_)
    fifo.push_back(QueueEntry{.allocation = std::move(name_), .will_free_at = current_seq_id});
  name_ = std::move(allocation);
  name_suballocation_ = suballocation;
  owned_by_command_list_ = owned_by_command_list;
}

void
DynamicBuffer::recycle(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (owned_by_command_list_) {
    if (name_.ptr() == allocation.ptr()) {
      owned_by_command_list_ = false;
      auto _ = std::move(allocation);
      return;
    }
  }
  fifo.push_back(QueueEntry{.allocation = std::move(allocation), .will_free_at = current_seq_id});
}

uint32_t
DynamicBuffer::nextSuballocation() {
  if (name_->hasSuballocatoin(name_suballocation_ + 1)) {
    return ++name_suballocation_;
  }
  return 0;
}

DynamicLinearTexture::DynamicLinearTexture(Texture *texture, Flags<TextureAllocationFlag> flags) :
    texture(texture),
    flags_(flags),
    name_(texture->current()) {}

void
DynamicLinearTexture::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
DynamicLinearTexture::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

Rc<TextureAllocation>
DynamicLinearTexture::allocate(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  Rc<TextureAllocation> ret;
  for (;;) {
    if (fifo.empty()) {
      break;
    }
    auto entry = fifo.front();
    if (entry.will_free_at > coherent_seq_id) {
      break;
    }
    ret = std::move(entry.allocation);
    fifo.pop_front();
    trimSpare(fifo, coherent_seq_id);
    break;
  }
  if (!ret.ptr())
    ret = texture->allocate(flags_);
  return ret;
}

void
DynamicLinearTexture::updateImmediateName(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation, bool owned_by_command_list) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (!owned_by_command_list_)
    fifo.push_back(QueueEntry{.allocation = std::move(name_), .will_free_at = current_seq_id});
  name_ = std::move(allocation);
  owned_by_command_list_ = owned_by_command_list;
}

void
DynamicLinearTexture::recycle(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (owned_by_command_list_) {
    if (name_.ptr() == allocation.ptr()) {
      owned_by_command_list_ = false;
      auto _ = std::move(allocation);
      return;
    }
  }
  fifo.push_back(QueueEntry{.allocation = std::move(allocation), .will_free_at = current_seq_id});
}

} // namespace dxmt
