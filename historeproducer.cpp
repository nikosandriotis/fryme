#include <iostream>
#include <CL/sycl.hpp>
#include "kernel.h"


int main(){

    //we need inside the gpu: 
    //  yy, moduleStart, id
    
    std::vector<uint16_t> yy;
    std::ifstream inputFile("yyinputGPU.txt");        // Input file stream object
        int current_number = 0;
        while (inputFile >> current_number){
            yy.push_back(current_number);
        }

        // Close the file.
        inputFile.close();
        
    std::vector<uint32_t> moduleStart;
    std::ifstream inputFile1("moduleStartGPU.txt");        // Input file stream object
        int current_number1 = 0;
        while (inputFile1 >> current_number1){
            moduleStart.push_back(current_number1);
        }

        // Close the file.
        inputFile1.close();
        
    std::vector<uint16_t> moduleInd;
    std::ifstream inputFile2("moduleindinput.txt");        // Input file stream object
        int current_number2 = 0;
        while (inputFile2 >> current_number2){
            moduleInd.push_back(current_number2);
        }

        // Close the file.
        inputFile2.close();
        
    auto queue = sycl::queue{sycl::gpu_selector()};
        
    auto d_yy = sycl::malloc_device<uint16_t>(48316, queue);
    auto d_moduleStart = sycl::malloc_device<uint32_t>(1790, queue);
    auto d_moduleInd = sycl::malloc_device<uint16_t>(48316, queue);    
    
    queue.memcpy(d_yy, yy.data(), 48316 * sizeof(uint16_t)).wait();
    queue.memcpy(d_moduleStart, moduleStart.data(), 1790 * sizeof(uint32_t)).wait();
    queue.memcpy(d_moduleInd, moduleInd.data(), 48316 * sizeof(uint16_t)).wait();
    
    int threadsPerBlock = 32;
    int blocks = 2000;
    
    constexpr uint32_t maxPixInModule = 4000;
    constexpr auto nbins = 418;
    using Hist = HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
    
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
                    local_mSize_acc(sycl::range<1>(sizeof(int)), cgh);
      sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
                    hist_acc(sycl::range<1>(sizeof(Hist)), cgh);
      sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
                    local_ws_acc(sycl::range<1>(sizeof(uint32_t) * 32), cgh);
      sycl::stream out(300000, 4096, cgh);
           
      auto digis_y_kernel     = d_yy;
      auto digis_ind_kernel   = d_moduleInd;
      auto clusters_s_kernel  = d_moduleStart;
      
      cgh.parallel_for(
          sycl::nd_range<1>(32, 32),
          [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {
            kernel(out, 
                   item, 
                   digis_y_kernel, 
                   digis_ind_kernel, 
                   clusters_s_kernel,
                   (int*)local_mSize_acc.get_pointer(), 
                   (Hist*)hist_acc.get_pointer(),
                   (uint32_t *)local_ws_acc.get_pointer()
		  );

      });
    }).wait();
    
    sycl::free(d_yy,queue);
    sycl::free(d_moduleStart,queue);
    sycl::free(d_moduleInd,queue);

}
