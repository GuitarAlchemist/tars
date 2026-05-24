#include <stdio.h>
#include <dlfcn.h>

// Function pointer types
typedef int (*tars_cuda_device_count_func)(void);
typedef int (*tars_cuda_init_func)(int);
typedef int (*tars_cuda_cleanup_func)(void);

int main() {
    printf("🧪 Testing TARS CUDA Library\n");
    printf("============================\n\n");
    
    // Load the library
    void* handle = dlopen("./libTarsCudaKernels.so", RTLD_LAZY);
    if (!handle) {
        printf("❌ Failed to load library: %s\n", dlerror());
        return 1;
    }
    printf("✅ Library loaded successfully\n");
    
    // Get function pointers
    tars_cuda_device_count_func device_count = (tars_cuda_device_count_func)dlsym(handle, "tars_cuda_device_count");
    tars_cuda_init_func cuda_init = (tars_cuda_init_func)dlsym(handle, "tars_cuda_init");
    tars_cuda_cleanup_func cuda_cleanup = (tars_cuda_cleanup_func)dlsym(handle, "tars_cuda_cleanup");
    
    if (!device_count || !cuda_init || !cuda_cleanup) {
        printf("❌ Failed to get function pointers: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    printf("✅ Function pointers loaded\n");
    
    // Test device count
    int count = device_count();
    printf("📊 CUDA devices found: %d\n", count);
    
    if (count > 0) {
        // Test initialization
        printf("🔧 Initializing CUDA on device 0...\n");
        int result = cuda_init(0);
        if (result == 0) {
            printf("✅ CUDA initialization successful\n");
            
            // Test cleanup
            printf("🧹 Cleaning up CUDA...\n");
            result = cuda_cleanup();
            if (result == 0) {
                printf("✅ CUDA cleanup successful\n");
            } else {
                printf("❌ CUDA cleanup failed: %d\n", result);
            }
        } else {
            printf("❌ CUDA initialization failed: %d\n", result);
        }
    } else {
        printf("⚠️ No CUDA devices found (may be normal in WSL)\n");
    }
    
    // Close library
    dlclose(handle);
    printf("\n🎉 TARS CUDA Library test complete!\n");
    
    return 0;
}
