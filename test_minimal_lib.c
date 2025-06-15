#include <stdio.h>
#include <dlfcn.h>

typedef int (*minimal_cuda_device_count_func)(void);
typedef int (*minimal_cuda_init_func)(int);

int main() {
    printf("Testing minimal CUDA library...\n");
    
    void* handle = dlopen("./libminimal_cuda.so", RTLD_LAZY);
    if (!handle) {
        printf("Error loading library: %s\n", dlerror());
        return 1;
    }
    
    minimal_cuda_device_count_func device_count = (minimal_cuda_device_count_func)dlsym(handle, "minimal_cuda_device_count");
    if (!device_count) {
        printf("Error getting device_count function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    
    minimal_cuda_init_func cuda_init = (minimal_cuda_init_func)dlsym(handle, "minimal_cuda_init");
    if (!cuda_init) {
        printf("Error getting cuda_init function: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    
    printf("Calling minimal_cuda_device_count...\n");
    int count = device_count();
    printf("Returned device count: %d\n", count);
    
    if (count > 0) {
        printf("Calling minimal_cuda_init(0)...\n");
        int result = cuda_init(0);
        printf("Init result: %d\n", result);
    }
    
    dlclose(handle);
    return 0;
}
