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

/*
 * Patch v2: idle sweep. trimSpare above only runs when the game maps a resource again, so a
 * resource that got many copies in a burst and was then left alone kept them all. Every dynamic
 * resource is on a global list; the queue's finishing thread walks it now and then.
 *
 * Sequence ids belong to one command queue. A resource records the queue it was last mapped
 * through (queue_tag_) and is only trimmed by that queue, so two devices never mix their ids.
 * A resource never mapped has no spare copies and no tag.
 *
 * Locks: list lock, then the resource's own lock (try only; a busy resource is skipped). The
 * released copies are destroyed after both locks are dropped. A destructor takes the list lock
 * to unlink, so a resource cannot go away while the sweep looks at it.
 */
uint64_t
idleAgeLimit() {
  static const uint64_t value = [] {
    auto s = env::getEnvVar("DXMT_DYNAMIC_IDLE_AGE");
    if (s.empty())
      return uint64_t(600);
    long long v = std::atoll(s.c_str());
    return v > 0 ? uint64_t(v) : uint64_t(0);
  }();
  return value;
}

constexpr uint64_t kIdleSweepInterval = 128;

template <typename T> struct IdleList {
  dxmt::mutex mutex;
  T *head = nullptr;

  void
  link(T *obj) {
    std::lock_guard<dxmt::mutex> lock(mutex);
    obj->idle_prev_ = nullptr;
    obj->idle_next_ = head;
    if (head)
      head->idle_prev_ = obj;
    head = obj;
  }

  void
  unlink(T *obj) {
    std::lock_guard<dxmt::mutex> lock(mutex);
    if (obj->idle_prev_)
      obj->idle_prev_->idle_next_ = obj->idle_next_;
    else
      head = obj->idle_next_;
    if (obj->idle_next_)
      obj->idle_next_->idle_prev_ = obj->idle_prev_;
    obj->idle_prev_ = obj->idle_next_ = nullptr;
  }
};

IdleList<DynamicBuffer> g_idle_buffers;
IdleList<DynamicLinearTexture> g_idle_textures;

template <typename Fifo, typename Out>
void
collectIdle(Fifo &fifo, uint64_t coherent_seq_id, uint64_t limit, Out &out) {
  while (!fifo.empty()) {
    auto &front = fifo.front();
    if (front.will_free_at > coherent_seq_id || coherent_seq_id - front.will_free_at <= limit)
      break;
    out.push_back(std::move(front.allocation));
    fifo.pop_front();
  }
}

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
    name_(buffer->current()) {
  g_idle_buffers.link(this);
}

DynamicBuffer::~DynamicBuffer() {
  g_idle_buffers.unlink(this);
}

void
DynamicBuffer::trimIdle(
    const void *queue_tag, uint64_t coherent_seq_id, uint64_t limit, std::vector<Rc<BufferAllocation>> &out
) {
  if (!mutex_.try_lock())
    return;
  if (queue_tag_ == queue_tag)
    collectIdle(fifo, coherent_seq_id, limit, out);
  mutex_.unlock();
}

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
DynamicBuffer::allocate(uint64_t coherent_seq_id, const void *queue_tag) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  queue_tag_ = queue_tag;
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
    name_(texture->current()) {
  g_idle_textures.link(this);
}

DynamicLinearTexture::~DynamicLinearTexture() {
  g_idle_textures.unlink(this);
}

void
DynamicLinearTexture::trimIdle(
    const void *queue_tag, uint64_t coherent_seq_id, uint64_t limit, std::vector<Rc<TextureAllocation>> &out
) {
  if (!mutex_.try_lock())
    return;
  if (queue_tag_ == queue_tag)
    collectIdle(fifo, coherent_seq_id, limit, out);
  mutex_.unlock();
}

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
DynamicLinearTexture::allocate(uint64_t coherent_seq_id, const void *queue_tag) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  queue_tag_ = queue_tag;
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

void
TrimIdleDynamicResources(const void *queue_tag, uint64_t coherent_seq_id) {
  uint64_t limit = idleAgeLimit();
  if (!limit || (coherent_seq_id % kIdleSweepInterval) != 0)
    return;
  std::vector<Rc<BufferAllocation>> buffers;
  std::vector<Rc<TextureAllocation>> textures;
  {
    std::lock_guard<dxmt::mutex> lock(g_idle_buffers.mutex);
    for (auto obj = g_idle_buffers.head; obj; obj = obj->idle_next_)
      obj->trimIdle(queue_tag, coherent_seq_id, limit, buffers);
  }
  {
    std::lock_guard<dxmt::mutex> lock(g_idle_textures.mutex);
    for (auto obj = g_idle_textures.head; obj; obj = obj->idle_next_)
      obj->trimIdle(queue_tag, coherent_seq_id, limit, textures);
  }
  // `buffers` and `textures` release the copies here, with no lock held
}

} // namespace dxmt
