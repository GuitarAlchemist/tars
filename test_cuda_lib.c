#include <stdio.h>
#include <dlfcn.h>

int main() {
    // Load the CUDA library
    void* handle = dlopen("./libminimal_cuda.so", RTLD_LAZY);
    if (!handle) {
        printf("Cannot load library: %s\n", dlerror());
        return 1;
    }
    
    // Get the device count function
    int (*device_count_func)() = dlsym(handle, "minimal_cuda_device_count");
    if (!device_count_func) {
        printf("Cannot find function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    
    // Call the function
    printf("Calling minimal_cuda_device_count...\n");
    int count = device_count_func();
    printf("Device count returned: %d\n", count);
    
    dlclose(handle);
    return 0;
}
