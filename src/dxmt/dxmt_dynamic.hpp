#pragma once
#include "dxmt_buffer.hpp"
#include "dxmt_texture.hpp"
#include <deque>
#include <vector>

namespace dxmt {

class DynamicBuffer {
public:
  void incRef();
  void decRef();

  Rc<BufferAllocation> allocate(uint64_t coherent_seq_id, const void *queue_tag);
  void updateImmediateName(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation, uint32_t suballocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation);
  uint32_t nextSuballocation();

  Rc<BufferAllocation>
  immediateName() {
    return name_;
  }

  uint32_t
  immediateSuballocation() {
    return name_suballocation_;
  }

  void *
  immediateMappedMemory() {
    return name_->mappedMemory(name_suballocation_);
  }

  DynamicBuffer(Buffer *buffer, Flags<BufferAllocationFlag> flags);
  ~DynamicBuffer();

  /* XoM low-memory patch v2: see TrimIdleDynamicResources */
  void trimIdle(const void *queue_tag, uint64_t coherent_seq_id, uint64_t limit, std::vector<Rc<BufferAllocation>> &out);
  DynamicBuffer *idle_prev_ = nullptr;
  DynamicBuffer *idle_next_ = nullptr;

  struct QueueEntry {
    Rc<BufferAllocation> allocation;
    uint64_t will_free_at;
  };

  /**
   * readonly
   */
  Buffer *buffer;

private:
  Flags<BufferAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  std::deque<QueueEntry> fifo;
  dxmt::mutex mutex_;
  Rc<BufferAllocation> name_;
  uint32_t name_suballocation_ = 0;
  bool owned_by_command_list_ = false;
  const void *queue_tag_ = nullptr; // the command queue whose sequence ids the FIFO holds; set by allocate()
};

class DynamicLinearTexture {
public:
  void incRef();
  void decRef();

  Rc<TextureAllocation> allocate(uint64_t coherent_seq_id, const void *queue_tag);
  void updateImmediateName(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation);

  Rc<TextureAllocation>
  immediateName() {
    return name_;
  };

  void *
  mappedMemory() {
    return name_->mappedMemory;
  }

  DynamicLinearTexture(Texture *buffer, Flags<TextureAllocationFlag> flags);
  ~DynamicLinearTexture();

  void trimIdle(const void *queue_tag, uint64_t coherent_seq_id, uint64_t limit, std::vector<Rc<TextureAllocation>> &out);
  DynamicLinearTexture *idle_prev_ = nullptr;
  DynamicLinearTexture *idle_next_ = nullptr;

  struct QueueEntry {
    Rc<TextureAllocation> allocation;
    uint64_t will_free_at;
  };
  /**
   * readonly
   */
  Texture *texture;

private:
  Flags<TextureAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  std::deque<QueueEntry> fifo;
  dxmt::mutex mutex_;
  Rc<TextureAllocation> name_;
  bool owned_by_command_list_ = false;
  const void *queue_tag_ = nullptr; // the command queue whose sequence ids the FIFO holds; set by allocate()
};

/**
 * XoM low-memory patch v2. Called by a command queue's finishing thread after sequence id
 * `coherent_seq_id` completed on the GPU. Every 128 ids it walks all dynamic resources that were
 * mapped through this queue and releases spare copies that have been free for more than
 * DXMT_DYNAMIC_IDLE_AGE ids (default 600, 0 = off). The copy in use is never touched.
 */
void TrimIdleDynamicResources(const void *queue_tag, uint64_t coherent_seq_id);

} // namespace dxmt
