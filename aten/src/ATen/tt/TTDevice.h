#pragma once

#include <ATen/core/ATen_fwd.h>
#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>

namespace tt::tt_metal {
class Buffer;
class IDevice;
}

namespace at::tt {

class TTAllocator : public c10::Allocator {
public:
  TTAllocator();

  virtual void emptyCache() const {}
  virtual void freeInactiveBuffers() const {}
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const { return -1; }
  virtual IntArrayRef getBufferShape(const void* ptr) const { return IntArrayRef(); }
  virtual id_t getBufferId(const void* ptr) const { return 0; }
  virtual void setBufferShape(const void* ptr, const IntArrayRef& shape)
    const {}
  virtual bool isSharedBuffer(const void* ptr) const { return false; }
  virtual bool isSharedStorageSupported() const { return false; }
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size)
    const { return DataPtr(); }
  virtual std::string formatSize(size_t size) const { return ""; }
  virtual void setLowWatermarkRatio(double ratio) const {}
  virtual void setHighWatermarkRatio(double ratio) const {}
  virtual ssize_t getLowWatermarkValue() const { return -1; }
  virtual size_t getLowWatermarkLimit() const { return 0; }
  virtual size_t getHighWatermarkLimit() const { return 0; }
  virtual size_t getTotalAllocatedMemory() const { return 0; }
  virtual size_t getCurrentAllocatedMemory() const { return 0; }
  virtual size_t getDriverAllocatedMemory() const { return 0; }
  virtual size_t getRecommendedMaxMemory() const { return 0; }
  virtual std::pair<const void*, uint32_t> getSharedBufferPtr(const void* ptr) const { return {nullptr, 0}; }
  virtual bool recordEvents(c10::ArrayRef<const void*> buffers) const { return false; }
  virtual bool waitForEvents(c10::ArrayRef<const void*> buffers) const {return false; }

  virtual DataPtr allocate(size_t n) override;
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override;

private:
  std::vector<std::shared_ptr<::tt::tt_metal::Buffer>> buffers_;
  ::tt::tt_metal::IDevice* device_;
};

TORCH_API TTAllocator* GetTTAllocator(bool useSharedAllocator = false);

} // namespace at::tt
