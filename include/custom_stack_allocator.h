#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

class CustomStackAllocator : public flatbuffers::Allocator {
 public:
  CustomStackAllocator(size_t alignment) : data_size_(0) {
    data_ = tflite::AlignPointerUp(data_backing_, alignment);
  }

  uint8_t* allocate(size_t size) override {
    TFLITE_DCHECK((data_size_ + size) <= kStackAllocatorSize);
    uint8_t* result = data_;
    data_ += size;
    data_size_ += size;
    return result;
  }

  void deallocate(uint8_t* p, size_t) override {}

  static CustomStackAllocator& instance(size_t alignment = 1) {
    // Avoid using true dynamic memory allocation to be portable to bare metal.
    static char inst_memory[sizeof(CustomStackAllocator)];
    static CustomStackAllocator* inst = new (inst_memory) CustomStackAllocator(alignment);
    return *inst;
  }

  static constexpr size_t kStackAllocatorSize = 8192;

 private:
  uint8_t data_backing_[kStackAllocatorSize];
  uint8_t* data_;
  int data_size_;

  void operator delete(void* p) {}
};
