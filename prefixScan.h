#include <CL/sycl.hpp>
#include <cstdint>

template <typename T>
void __forceinline warpPrefixScan(T* c, uint32_t i, uint32_t mask, sycl::nd_item<1> item, sycl::stream out) {
  
  auto x = c[i];
  
  auto  laneId = item.get_local_id(0) & 0x1f;
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = sycl::shift_group_right(item.get_sub_group(), x, offset); //FIXME_ it was __shfl_up_sync
    
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

    // same as above (1), may remove
    // limited to 32*32 elements....
    template <typename T>
    __forceinline void blockPrefixScan(T* c,
                                         uint32_t size,
                                         sycl::nd_item<1> item,
                                         T* ws, sycl::stream out) {
      
      
      const auto first = item.get_local_id(0);
      
      size_t idtest = item.get_sub_group().get_local_linear_id();
      uint32_t local_valtest = (first < size ? 1u : 0u) << idtest;
      
//       item.barrier();
//       for (auto i = first; i < size; i += item.get_local_range(0)) {
//       if(item.get_group(0)==0)
//       out << size << " ";
//       }
//      if (first==0)
//        out << "c=" << c << " ws=" << ws << "\n";
//      if (first==0)
//        out << "\n";
//      for (auto i = first; i < size; i += item.get_local_range(0)) {
//        out << c[i] << " ";
//      }
      auto mask = sycl::reduce_over_group(item.get_sub_group(), local_valtest, sycl::plus<>());
      for (auto i = first; i < size; i += item.get_local_range(0)) {
       
     // out << i / 32 << " ";	      
          //item.barrier();
//        if(item.get_group(0)==0)
          //out << c[i]<< " ";
       // item.barrier();
       // out << c[i] << " ";

       warpPrefixScan(c, i, mask, item, out);
        
        //item.barrier();
       // out << c[i] << " ";
       auto laneId = item.get_local_id(0) & 0x1f;      
       auto warpId = i / 32;
       if (31 == laneId){
         //out << i << ": " << c[64] << "\t";
         auto temp = c[i];
         ws[warpId] = temp;
         //out << ws+warpId <<  ": "<< ws[warpId] << "\n"; 
         //out << c[64] << "\n"; 
              }
            
        //out << ws[warpId] << " ";                  
        size_t idtest = item.get_sub_group().get_local_linear_id();
        uint32_t local_valtest = ((i + item.get_local_range(0)) < size ? 1u : 0u) << idtest;          
        mask = sycl::reduce_over_group(item.get_sub_group(), local_valtest, sycl::plus<>());
      }
      
//      if (first==0)
//        out << "\n";
//      for (auto i = first; i < size; i += item.get_local_range(0)) {
//        out << c[i] << " ";
//      }
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
      if (size <= 32)
        return;
      if (item.get_local_id(0) < 32)
        warpPrefixScan(ws, item.get_local_id(0), 0xffffffff, item, out);
      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);
//      
      for (auto i = first + 32; i < size; i += item.get_local_range(0)) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
////      //Same as above (0)
      item.barrier(sycl::access::fence_space::local_space);

      //__CUDA_ARCH__
      //for (uint32_t i = 1; i < size; ++i)
      //  c[i] += c[i - 1];

    }

