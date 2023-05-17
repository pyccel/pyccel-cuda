#ifndef HO_CUDA_NDARRAYS_H
# define HO_CUDA_NDARRAYS_H

#include "../ndarrays/ndarrays.h"

__global__
void cuda_array_arange_int8(t_ndarray arr, int start);
__global__
void cuda_array_arange_int32(t_ndarray arr, int start);
__global__
void cuda_array_arange_int64(t_ndarray arr, int start);
__global__
void cuda_array_arange_double(t_ndarray arr, int start);

__global__
void cuda_array_fill_int8(int8_t c, t_ndarray arr);
__global__
void cuda_array_fill_int32(int32_t c, t_ndarray arr);
__global__
void cuda_array_fill_int64(int64_t c, t_ndarray arr);
__global__
void cuda_array_fill_double(double c, t_ndarray arr);

t_ndarray   cuda_array_create(int32_t nd, int64_t *shape, enum e_types type, bool is_view, enum e_memory_locations location);
__device__
void    shared_array_init(t_ndarray *arr);
int32_t         cuda_free_array(t_ndarray dump);

int32_t cuda_free_host(t_ndarray arr);

__host__ __device__
int32_t cuda_free(t_ndarray arr);

__host__ __device__
int32_t cuda_free_pointer(t_ndarray arr);

#ifdef HO_CUDA_PYCCEL
__device__ inline void    shared_array_init(t_ndarray *arr)
{
    switch (arr->type)
    {
        case nd_int8:
            arr->type_size = sizeof(int8_t);
            break;
        case nd_int16:
            arr->type_size = sizeof(int16_t);
            break;
        case nd_int32:
            arr->type_size = sizeof(int32_t);
            break;
        case nd_int64:
            arr->type_size = sizeof(int64_t);
            break;
        case nd_float:
            arr->type_size = sizeof(float);
            break;
        case nd_double:
            arr->type_size = sizeof(double);
            break;
        case nd_bool:
            arr->type_size = sizeof(bool);
            break;
    }
    arr->length = 1;
    for (int32_t i = 0; i < arr->nd; i++)
        arr->length *= arr->shape[i];
    arr->buffer_size = arr->length * arr->type_size;
    for (int32_t i = 0; i < arr->nd; i++)
    {
        arr->strides[i] = 1;
        for (int32_t j = i + 1; j < arr->nd; j++)
            arr->strides[i] *= arr->shape[j];
    }
}

__device__
t_ndarray array_slicing(t_ndarray arr, int n, ...)
{
    t_ndarray view;
    va_list  va;
    t_slice slice;
    int32_t start = 0;
    int32_t j;

    view.nd = n;
    view.type = arr.type;
    view.type_size = arr.type_size;
    view.shape = malloc(sizeof(int64_t) * view.nd);
    view.strides = malloc(sizeof(int64_t) * view.nd);
    view.is_view = true;

    va_start(va, n);
    j = 0;
    for (int32_t i = 0; i < arr.nd; i++)
    {
        slice = va_arg(va, t_slice);
        if (slice.type == RANGE)
        {
            view.shape[j] = (slice.end - slice.start + (slice.step - 1)) / slice.step;
            view.strides[j] = arr.strides[i] * slice.step;
            j++;
        }
        start += slice.start * arr.strides[i];
    }
    va_end(va);

    view.raw_data = arr.raw_data + start * arr.type_size;
    view.length = 1;
    for (int32_t i = 0; i < view.nd; i++)
            view.length *= view.shape[i];
    return (view);
}

#endif
#undef HO_CUDA_PYCCEL
#endif