to reproduce the error run the following command:
```dpcpp historeproducer.cpp -o historeproducer```
```./historeproducer```

It will not work on GPU, but it will on CPU (change the device selector in ```historeproducer.cpp``` and run again).
The way we found to fix it is to use ```sycl::ext::oneapi::group_local_memory_for_overwrite```. 
To do this uncomment lines from 38 to 41 in ```kernel.h``` and change the name of the variables passed to the kernel (lines 10-11, just add a letter to the names) not to break the program.

What gets printed are same of the offsets of hist, that is one of the two accessors.
