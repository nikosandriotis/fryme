#include <CL/sycl.hpp>

template <typename T,  
          sycl::access::address_space addrSpace,
          sycl::memory_scope Scope,
          sycl::memory_order memOrder = sycl::memory_order::relaxed>
          
      inline T atomic_fetch_add(T* addr, T operand){
      auto atm = 
            sycl::atomic_ref<T, memOrder, Scope, 
                                  addrSpace>(addr[0]);
      return atm.fetch_add(operand);
    }
    
template <typename T,  
          sycl::access::address_space addrSpace,
          sycl::memory_scope Scope,
          sycl::memory_order memOrder = sycl::memory_order::relaxed>
    inline T atomic_fetch_min(T *addr, T operand) {
    auto atm =
              sycl::atomic_ref<T, memOrder, Scope,
                                                  addrSpace>
                              (addr[0]);            
    return atm.fetch_min(operand);
    }

 template <typename T,  
          sycl::access::address_space addrSpace,
          sycl::memory_scope Scope,
          sycl::memory_order memOrder = sycl::memory_order::relaxed>
    inline T atomic_fetch_sub(T *addr, T operand) {
          auto atm =
              sycl::atomic_ref<T, memOrder, Scope,
                                   addrSpace>(addr[0]);
      return atm.fetch_sub(operand);
    }
