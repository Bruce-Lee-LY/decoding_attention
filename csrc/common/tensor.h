// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: tensor

#pragma once

#include <random>

#include "common.h"

template <typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t> &shape, const std::string &name = "Tensor", float min = -1.0, float max = 1.0)
        : m_shape(shape), m_name(name), m_min(min), m_max(max) {
        DA_CHECK_GT(shape.size(), 0);
        for (size_t i = 0; i < shape.size(); ++i) {
            DA_CHECK_GT(shape[i], 0);
        }

        m_elem_num = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
        DA_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new T[m_elem_num];
        DA_CHECK(m_host_ptr);
        DA_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(T)));
        DA_CHECK(m_dev_ptr);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        for (size_t i = 0; i < m_elem_num; ++i) {
            m_host_ptr[i] = static_cast<T>(uniform(engine));
        }

        DA_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(T), cudaMemcpyHostToDevice));

        DLOG("%s: %zu, cpu: %p, gpu: %p", m_name.c_str(), m_elem_num, m_host_ptr, m_dev_ptr);
    }

    ~Tensor() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            DA_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    std::vector<size_t> getShape() const {
        return m_shape;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    T *getHostPtr() const {
        return m_host_ptr;
    }

    T *getDevPtr() const {
        return m_dev_ptr;
    }

    void tearUp(Tensor<T> *base) {
        DA_CHECK(base);
        DA_CHECK_EQ(m_elem_num, base->getElemNum());

        DA_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void moveToHost() {
        DA_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        DA_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(T));
    }

    void memSetDevice() {
        DA_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(T)));
    }

    void checkValue(Tensor<T> *base) {
        DA_CHECK(base);
        DA_CHECK_EQ(m_elem_num, base->getElemNum());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(
                std::abs(static_cast<float>(m_host_ptr[i]) - static_cast<float>(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        DLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const std::vector<size_t> m_shape;
    const std::string m_name = "Tensor";
    // the threshold of the random tensor will affect the difference of the dmha results
    const float m_min = -1.0;
    const float m_max = 1.0;

    size_t m_elem_num = 0;
    T *m_host_ptr = nullptr;
    T *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    DA_DISALLOW_COPY_AND_ASSIGN(Tensor);
};
