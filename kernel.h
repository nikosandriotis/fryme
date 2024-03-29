#include "atomics.h"
#include "histocontainer.h"

constexpr uint32_t maxPixInModule = 4000;
constexpr auto nbins = 418;
using Hist = HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

void kernel(sycl::stream out, sycl::nd_item<1> item, uint16_t const* __restrict__ y,
            uint16_t const* __restrict__ id,  uint32_t const* __restrict__ moduleStart, int* msize,
                Hist* hist, 
                Hist::Counter* ws
		){
             
      if (item.get_group(0) >= moduleStart[0])
      return;
      
      auto firstPixel = moduleStart[1 + item.get_group(0)];
      auto thisModuleId = id[firstPixel];
      
      auto first = firstPixel + item.get_local_id(0);
      
      *msize = 48316;
      item.barrier();
      
      // skip threads not associated to an existing pixel
      for (int i = first; i < 48316; i += item.get_local_range(0)) {
        if (id[i] == 9999)  // skip invalid pixels
          continue;
        if (id[i] != thisModuleId) {  //find the first pixel in a different module
          atomic_fetch_min<int, 
                                              sycl::access::address_space::local_space, 
                                              sycl::memory_scope::work_group>
                                              (static_cast<int*>(msize), static_cast<int>(i));
          break;
        }
      }
      
      // auto wsbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[32]>(item.get_group());
      // uint32_t* ws = (uint32_t*)wsbuff.get();
      // auto histbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<Hist>(item.get_group());
      // Hist* hist = (Hist*)histbuff.get();
    
      for (auto j = item.get_local_id(0); j < Hist::totbins(); j += item.get_local_range(0)) {
        hist->off[j] = 0;
      }
      item.barrier();
      
      constexpr int maxPixInModule = 4000;
      if (0 == item.get_local_id(0)) {
        if ((*msize - static_cast<int>(firstPixel)) > maxPixInModule) {
          out << "too many pixels in module " << thisModuleId << ": " << *msize - static_cast<int>(firstPixel) << " > "
              << maxPixInModule << "\n";
          *msize = maxPixInModule + firstPixel;
        }
      }
      item.barrier();
      
      for (int i = first; i < *msize; i += item.get_local_range(0)) {
        if (id[i] == 9999) {  // skip invalid pixels
          continue;
        }
        hist->count(y[i]);
      }
      item.barrier();
      

      if (item.get_local_id(0) < 32u)
        ws[item.get_local_id(0)] = 0; 
      
    item.barrier();
      hist->finalize(item, out, ws);
      item.barrier();
     
    for (auto j = 0; j < (int)Hist::totbins(); j ++) {
    if (id[j] == 9999 && j >= (int)Hist::totbins()) {  // skip invalid pixels
        continue;
      }      
      sum+=hist->off[j];
    }
    
    item.barrier();
    
    if (item.get_local_id(0)==0){
     out << "Offest sum: " << sum << "\n";
     }
      
}
