#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>

typedef float half;
#define Num 3

namespace wmma{

   struct row_major;
   struct col_major;
   struct matrix_a;
   struct matrix_b;
   struct accumulator;
   enum layout_t { mem_row_major, mem_col_major };

   inline __device__ void fill_fragment(float* c,float b){
       
      return;
   }

    inline __device__ void load_matrix_sync(float* a_frag , half *a ,int test ){
      for (int i = 0; i < Num * Num; i++) {
        a_frag[i] = a[i];  
    }
      return;
   }

   inline __device__  void mma_sync(float* c_frag , float*  a_frag ,float* b_frag) {
     for (int i = 0; i < Num; i++) {
        for (int j = 0; j < Num; j++) {
            c_frag[i * Num + j] = 0.0;
            for (int k = 0; k < Num; k++) {
                c_frag[i * Num + j] += a_frag[i * Num + k] * b_frag[k * Num + j];
            }
        }
    }
     return;
   }

    inline __device__  void store_matrix_sync(float *c , float*  c_frag , int cda , layout_t layout ){
      for (int i = 0; i < Num * Num; i++) {
           c[i] = c_frag[i];
      } 
      return ;
   }
}
