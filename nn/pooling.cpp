#include <cstdlib>
#include <cstdio>
#include <iostream>

template <typename T>
inline void _do_pooling(int batch_size, int n_output, int n_height, int n_width,
                        int pool_h, int pool_w, int ret_h, int ret_w,
                        const T *after_filter, T *ret, T *M) {
  for (int n = 0; n < batch_size; n++)
    for (int c = 0; c < n_output; c++) {
      int before_output = n * n_output + c;
      int before_ret_pixel = before_output * ret_h * ret_w;
      int before_output_pixel = before_output * n_height * n_width;
      for (int ph = 0; ph < ret_h; ph++)
        for (int pw = 0; pw < ret_w; pw++) {
          int hstart = ph * pool_h;
          int wstart = pw * pool_w;
          int hend = hstart + pool_h;
          int wend = wstart + pool_w;
          int best_h = hstart;
          int best_w = wstart;
          int begin = before_output_pixel + hstart * n_width;
          T best = *(after_filter + begin + wstart);

          for (int h = hstart; h < hend; h++) {
            for (int w = wstart; w < wend; w++) {
              const T *now = after_filter + begin + w;
              if (*now > best) {
                best = *now;
                best_h = h;
                best_w = w;
              }
            }
            begin += n_width;
          }
          *(ret + before_ret_pixel + ph * ret_w + pw) = best;
          *(M + before_output_pixel + best_h * n_width + best_w) = 1.0;
        }
    }
}

template <typename T>
inline void _upsample(int batch_size, int n_output, int n_height, int n_width,
                      int pool_h, int pool_w, int ret_h, int ret_w,
                      T *error_before_pooling, const T *error_output) {
  for (int n = 0; n < batch_size; n++)
    for (int c = 0; c < n_output; c++) {
      T *before_output = error_before_pooling + (n * n_output + c) * n_height * n_width;
      const T *error_before_output = error_output + (n * n_output + c) * ret_h * ret_w;
      for (int ph = 0; ph < n_height; ph++) {
        T *before_output_pixel = before_output + ph * n_width;
        const T *error_output_now = error_before_output + ph / pool_h * ret_w;
        for (int pw = 0; pw < n_width; pw++) {
          T *now = before_output_pixel + pw;
          if (*now == 1.0)
            *now = *(error_output_now + pw/pool_w);
        }
      }
    }
}

extern "C" {
  void do_pooling(int len, int batch_size, int n_output, int n_height, int n_width,
                  int pool_h, int pool_w, int ret_h, int ret_w,
                  const void *after_filter, void *ret, void *M) {
    switch(len) {
      case sizeof(float):
        _do_pooling<float>(batch_size, n_output, n_height, n_width,
                           pool_h, pool_w, ret_h, ret_w,
                           (const float*)after_filter,
                           (float*)ret, (float*)M);
        break;
      case sizeof(double):
        _do_pooling<double>(batch_size, n_output, n_height, n_width,
                            pool_h, pool_w, ret_h, ret_w,
                            (const double*)after_filter,
                            (double*)ret, (double*)M);
        break;
      default:
        fprintf(stderr, "Unknown len: %d\n", len);
        exit(1);
    }
  }
}

extern "C" {
  void upsample(int len, int batch_size, int n_output, int n_height, int n_width,
                  int pool_h, int pool_w, int ret_h, int ret_w,
                  void *error_before_pooling, const void *error_output) {
    switch(len) {
      case sizeof(float):
        _upsample<float>(batch_size, n_output, n_height, n_width,
                           pool_h, pool_w, ret_h, ret_w,
                           (float*) error_before_pooling, (const float*)error_output);
        break;
      case sizeof(double):
        _upsample<double>(batch_size, n_output, n_height, n_width,
                           pool_h, pool_w, ret_h, ret_w,
                           (double*) error_before_pooling, (const double*)error_output);
        break;
      default:
        fprintf(stderr, "Unknown len: %d\n", len);
        exit(1);
    }
  }
}
