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


};