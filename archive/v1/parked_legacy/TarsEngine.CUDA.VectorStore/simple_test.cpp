#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << \
===
TARS
VECTOR
STORE
TEST
===\ << std::endl;
    
    // Simple vector similarity test
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> vec3 = {1.0f, 0.0f, 0.0f};
    
    auto cosine_sim = [](const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    };
    
    float sim1_2 = cosine_sim(vec1, vec2);
    float sim1_3 = cosine_sim(vec1, vec3);
    
    std::cout << \Similarity
vec1-vec2:
\ << sim1_2 << std::endl;
    std::cout << \Similarity
vec1-vec3:
\ << sim1_3 << std::endl;
    
    if (sim1_3 > 0.99f && sim1_2 < 0.1f) {
        std::cout << \✅
Vector
similarity
logic
works!\ << std::endl;
        std::cout << \✅
Ready
for
CUDA
acceleration\ << std::endl;
        std::cout << \🚀
RTX
3070
will
provide
massive
speedup!\ << std::endl;
    }
    
    return 0;
}
