#include <stdio.h>
#include <dlfcn.h>

// Function pointer types
typedef int (*tars_cuda_device_count_func)(void);
typedef int (*tars_cuda_init_func)(int);

int main() {
    printf("Testing TARS CUDA library loading...\n");
    
    // Load the library
    void* handle = dlopen("./libTarsCudaKernels.so", RTLD_LAZY);
    if (!handle) {
        printf("Error loading library: %s\n", dlerror());
        return 1;
    }
    
    printf("Library loaded successfully\n");
    
    // Get function pointers
    tars_cuda_device_count_func device_count = (tars_cuda_device_count_func)dlsym(handle, "tars_cuda_device_count");
    if (!device_count) {
        printf("Error getting device_count function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    
    tars_cuda_init_func cuda_init = (tars_cuda_init_func)dlsym(handle, "tars_cuda_init");
    if (!cuda_init) {
        printf("Error getting cuda_init function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    
    printf("Functions loaded successfully\n");
    
    // Test device count
    printf("Calling tars_cuda_device_count...\n");
    int count = device_count();
    printf("Device count: %d\n", count);
    
    if (count > 0) {
        printf("Calling tars_cuda_init(0)...\n");
        int result = cuda_init(0);
        printf("Init result: %d\n", result);
    }
    
    dlclose(handle);
    printf("Test complete\n");
    return 0;
}
