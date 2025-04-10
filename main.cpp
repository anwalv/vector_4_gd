#include <iostream>
#include <immintrin.h>
#include <cmath>

struct vector4 {
private:
    __m128 data;

public:
    vector4(float x, float y, float z) {
        data = _mm_set_ps(0.0f, z, y, x);
    }

    vector4(float x, float y, float z, float w) {
        data = _mm_set_ps(w, z, y, x);
    }

    float x() const {
        return _mm_cvtss_f32(data);
    }

    float y() const {
        return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(1, 1, 1, 1)));
    }

    float z() const {
        return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(2, 2, 2, 2)));
    }

    float w() const {
        return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(3, 3, 3, 3)));
    }

    vector4& add(const vector4 &other) {
        data = _mm_add_ps(data, other.data);
        return *this;
    }

    vector4& add(float x, float y, float z) {
        __m128 other = _mm_set_ps(0.0f, z, y, x);
        data = _mm_add_ps(data, other);
        return *this;
    }

    vector4& sub(const vector4 &other) {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }

    vector4& sub(float x, float y, float z) {
        __m128 other = _mm_set_ps(0.0f, z, y, x);
        data = _mm_sub_ps(data, other);
        return *this;
    }

    vector4& mul(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }

    vector4& mul(float scale, float w_scale) {
        __m128 scalar = _mm_set_ps(w_scale, scale, scale, scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }

    vector4& div(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }

    vector4& div(float scale, float w_scale) {
        __m128 scalar = _mm_set_ps(w_scale, scale, scale, scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }

    vector4& dot(const vector4 &other) {
        __m128 mul = _mm_mul_ps(data, other.data);
        __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(mul, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        __m128 dotSingleValue = _mm_add_ss(sums, shuf);
        data = _mm_shuffle_ps(dotSingleValue, dotSingleValue, 0);
        return *this;
    }

    vector4& dot(float x, float y, float z) {
        vector4 other(x, y, z);
        return dot(other);
    }
    void print(const std::string& label = "") const {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        std::cout << "(" << x() << ", " << y() << ", " << z() << ", " << w() << ")" << std::endl;
    }

};
int main() {
    //------------------------------------------
    std::cout << "===Test1===" << std::endl;
    vector4 v1(1.0f, 2.0f, 3.0f);
    vector4 v2(4.0f, 5.0f, 6.0f, 1.0f);

    v1.print("v1");
    v2.print("v2");

    //------------------------------------------
    std::cout << "\n=== Test2 ===" << std::endl;
    vector4 v3 = v1;
    v3.add(v2);
    v3.print("v1 + v2");

    vector4 v4 = v1;
    v4.add(2.0f, 3.0f, 4.0f);
    v4.print("v1 + (2,3,4)");

    //------------------------------------------
    std::cout << "\n=== Test3 ===" << std::endl;
    vector4 v5 = v2;
    v5.sub(v1);
    v5.print("v2 - v1");

    vector4 v6 = v2;
    v6.sub(1.0f, 1.0f, 1.0f);
    v6.print("v2 - (1,1,1)");

    //------------------------------------------
    std::cout << "\n===Test4===" << std::endl;
    vector4 v7 = v1;
    v7.mul(2.0f);
    v7.print("v1 * 2");

    vector4 v8 = v2;
    v8.mul(3.0f, 0.5f);
    v8.print("v2 * (3,3,3,0.5)");

    //------------------------------------------
    std::cout << "\n=== Test 5 ===" << std::endl;
    vector4 v9 = v1;
    v9.div(2.0f);
    v9.print("v1 / 2");

    vector4 v10 = v2;
    v10.div(2.0f, 0.5f);
    v10.print("v2 / (2,2,2,0.5)");
    //------------------------------------------
    std::cout << "\n=== Test 6 ===" << std::endl;
    vector4 v11 = v1;
    v11.dot(v2);
    v11.print("v1 * v2");

    vector4 v12 = v2;
    v12.dot(1.0f, 2.0f, 3.0f);
    v12.print("v2 * (1,2,3)");

    return 0;
}