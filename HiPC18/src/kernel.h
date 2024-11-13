#pragma once

// with VWARP more K
__global__ void comp_kernel_COO(int const *__restrict__ row_ind,
                                int const *__restrict__ col_ind,
                                float *matrixP,
                                const float *__restrict__ matrixA,
                                const float *__restrict__ matrixB,
                                int nnz,
                                int n_rows,
                                int n_cols,
                                int k,
                                int tile_stIdx, //
                                int tile_limIdx,
                                int *d_last_blockIdx,
                                int *active_row,
                                int tile_no,
                                int t_st,
                                int act_rlimit, // 活跃行的限制; 矩阵A的行数
                                int sh_tile, // 默认输入为192
                                int k_slc) {
    unsigned int tId = threadIdx.x; // 在当前线程块中的线程ID

    // 如果tid是偶数, laneID则是0; 如果tid是奇数, laneID则是1. 在后面的代码中作用在确保不同线程尅并行处理不同的数据段, 避免了线程之间的数据竞争
    unsigned int laneId = tId & 1;

//    unsigned int c = (blockIdx.x * blockDim.x + tId); // 唯一线程id, 结论是一个线程计算一个结果, c代表当前线程处理的结果矩阵的位置.
    int block_st = 0; // 用来指示当前线程块从哪个位置开始处理数据.保证每个线程块处理的数据区间是唯一的.
    if (blockIdx.x == 0) {
        block_st = tile_stIdx;
    } else {
        block_st = d_last_blockIdx[blockIdx.x - 1];
    }
    int block_lim = d_last_blockIdx[blockIdx.x]; // 与block_st对应的结束索引

    __shared__ float sh_r[32 * 192]; // 共享内存储存密集矩阵A的切片数据
    int WARP_ID = tId >> 5; // 得到当前线程在第几个线程束中.
    int tid_in_WARP = tId & 31; // 表示线程在线程束内的索引. (如果tid为5(0101), tid & 31 的结果就是0101 & 11111 = 0101(5))
    int WARP_SIZE = 32; // 线程束大小

    int step = blockDim.x >> 5; // 每个线程束在处理数据时跨越的间隔

    int t = tid_in_WARP; // 表示线程在线程束内的索引

    // sh_tile : 每个线程束应该处理的元素数量. (blockIdx.x * sh_tile + i) < act_rlimit：确保当前线程束处理的元素总数没有超出活动行的限制
    for (int i = WARP_ID; i < sh_tile && (blockIdx.x * sh_tile + i) < act_rlimit; i += step) {
        for (int w_r = 0; w_r < k_slc; w_r += WARP_SIZE)
            sh_r[i * k_slc + t + w_r] = matrixA[active_row[blockIdx.x * sh_tile + i] * k + t + t_st + w_r];
    }

    __syncthreads(); // 加载数据到共享内存中必须地同步操作.

    // (tId >> 1)将线程的索引右移1位，相当于除以2，这可能是为了在每次迭代中处理两个元素，从而提高效率.
    // c < block_lim 保证线程不会超出处理元素的区间.
    for (int c = block_st + (tId >> 1); c < block_lim; c += (blockDim.x >> 1)) {
//        float sm = 0, g = 0;
        float sm1 = 0, sm2 = 0;
        int row = row_ind[c];
        int col = col_ind[c];
        int sh_row = row - blockIdx.x * sh_tile; // 计算当前行在共享内存中的相对位置.

        //for (int t = laneId*16; t < (laneId+1)*16; t+=8){
        // 每次处理8个元素, 这与float4数据类型的向量化操作相匹配. k_slc : 每个线程束处理密集矩阵的切片的元素数量
        for (int t = laneId * k_slc / 2; t < (laneId + 1) * k_slc / 2; t += 8) {
            float4 rtmp1 = *((float4 * ) & sh_r[sh_row * k_slc + t]);
            float4 ctmp1 = *((float4 * ) & matrixB[col * k + t_st + t]);
            sm1 += rtmp1.x * ctmp1.x + rtmp1.y * ctmp1.y + rtmp1.z * ctmp1.z + rtmp1.w * ctmp1.w;

            float4 rtmp2 = *((float4 * ) & sh_r[sh_row * k_slc + t + 4]);
            float4 ctmp2 = *((float4 * ) & matrixB[col * k + t_st + t + 4]);
            sm2 += rtmp2.x * ctmp2.x + rtmp2.y * ctmp2.y + rtmp2.z * ctmp2.z + rtmp2.w * ctmp2.w;
        }
        sm1 += __shfl_xor(sm1, 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程2的sm2上
        sm2 += __shfl_xor(sm2, 1);
//        sm1 += __shfl_xor_sync(0xFFFFFFFF, sm1, 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程2的sm2上
//        sm2 += __shfl_xor_sync(0xFFFFFFFF, sm2, 1);
        //val[c] = val[c] * (sm1 + sm2);
        matrixP[c] += (sm1 + sm2); // 将分来计算的两个元素加在一起储存到结果矩阵

    }
    __syncthreads();
}


// //Perfecto with VWARP
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, 
//     const float * __restrict__ u, const float * __restrict__ v, int nnz, int n_rows, int n_cols, int k, 
//     int tile_stIdx, int tile_limIdx, int *d_last_blockIdx,int *active_row, int tile_no, int tile_st)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int laneId = tId & 1;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int block_st = 0;
//     if(blockIdx.x==0) block_st = tile_stIdx;
//     else block_st=d_last_blockIdx[blockIdx.x-1];
//     int block_lim=d_last_blockIdx[blockIdx.x];
//     // if(blockIdx.x > 1660 && tId == 0)
//     //     printf("block lim %d %d %d\n", blockIdx.x, block_st, block_lim );

//     __shared__ float sh_r[32*180]; 
//     int sh_tile = 180;
//     int WARP_ID = tId >> 5;
//     int tid_in_WARP = tId & 31;
//     // for (int t = 0; t < k; ++t){
//     //     for (int i = tId; i < sh_tile && (i +(tile_no*sh_tile)) < n_rows; i+=blockDim.x){
//     //         sh_r[t * sh_tile + i] = u[ (t * n_rows) + active_row[blockIdx.x*sh_tile+i]];
//     //     }
//     // }    
//     int step = blockDim.x >> 5;
//     //for (int t = 0; t < k; ++t)
//     // {   int t = tid_in_WARP;
//     //     for (int i = WARP_ID; i < sh_tile && (i +(tile_no*sh_tile)) < n_rows; i+=step){
//     //         sh_r[t * sh_tile + i] = u[ (t * n_rows) + active_row[blockIdx.x*sh_tile+i]];
//     //     }
//     // } 
//     {   int t = tid_in_WARP;
//         for (int i = WARP_ID; i < sh_tile && (i +(tile_no*sh_tile)) < n_rows; i+=step){
//             sh_r[i *  k + t] = u[active_row[blockIdx.x*sh_tile+i] *k + t];
//         }
//     } 
//     __syncthreads();

//     for (int c = block_st + (tId >> 1); c < block_lim; c+=(blockDim.x >> 1)){
//         float sm =0 , g=0, sm1 =0, sm2=0;
//         int row = row_ind[c];
//         int col = col_ind[c];


//        // int sh_row = row - blockIdx.x * sh_tile;
//         int sh_row = row - blockIdx.x * sh_tile;
//         // if( blockIdx.x < 3 && blockIdx.x <5 && sh_row< 2)
//         //     printf("GPU %d %d :: %d %d %d %d %f\n",c , blockIdx.x, row, active_row[row], passive_row[row], sh_row, sh_r[ sh_row] );
//         //for (int t = laneId; t < k; t+=1)
//         // for (int t = 0; t < k; ++t)
//         // {
//         //     //sm += u[row*k+t] * v[col*k+t];
//         //    // sm += sh_r[t * sh_tile + sh_row] * v[t*n_cols+col];
//         //     //sm += sh_r[t * sh_tile + sh_row] * v[col*k+t];
//         //     g += sh_r[sh_row * k  + t] * v[col*k+t];
//         //     //g += sh_r[t * sh_tile + sh_row] * v[col*k+t];
//         //     //sm += u[t*n_rows+row] * v[t*n_cols+col];
//         // }
//        for (int t = laneId*16; t < (laneId+1)*16; t+=8)
//         //for (int t = 0; t < k; t+=8)
//         {
//             float4 rtmp1 = *((float4*) &sh_r[sh_row * k + t]); 
//             float4 ctmp1 = *((float4*) &v[col * k + t]);
//             sm1+= rtmp1.x * ctmp1.x + rtmp1.y * ctmp1.y +  rtmp1.z * ctmp1.z +  rtmp1.w * ctmp1.w ; 

//             float4 rtmp2 = *((float4*) &sh_r[sh_row * k + t+4]); 
//             float4 ctmp2 = *((float4*) &v[ col * k + t+4]);
//             sm2+= rtmp2.x * ctmp2.x + rtmp2.y * ctmp2.y +  rtmp2.z * ctmp2.z +  rtmp2.w * ctmp2.w ; 
//         }
//         sm1 += __shfl_xor(sm1, 1);
//         sm2 += __shfl_xor(sm2, 1);
//         val[c] = val[c] * (sm1 + sm2);

//         // __syncthreads();
//         // // g += __shfl_xor(g, 4);
//         // // g += __shfl_xor(g, 2);
//         // g += __shfl_xor(g, 1);

//         //__syncthreads();
//         //val[c] = val[c] * sm; 
//         //val[c] = val[c] * g; 

//     }  
// }




//without VWARP
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, 
//     const float * __restrict__ u, const float * __restrict__ v, int nnz, int n_rows, int n_cols, int k, 
//     int tile_stIdx, int tile_limIdx, int *d_last_blockIdx,int *active_row, int tile_no)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int laneId = tId & 1;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int block_st = 0;
//     if(blockIdx.x==0) block_st = tile_stIdx;
//     else block_st=d_last_blockIdx[blockIdx.x-1];
//     int block_lim=d_last_blockIdx[blockIdx.x];
//     // if(blockIdx.x > 1660 && tId == 0)
//     //     printf("block lim %d %d %d\n", blockIdx.x, block_st, block_lim );

//     __shared__ float sh_r[180*32]; 
//     int sh_tile = 180;
//     for (int t = 0; t < k; ++t){
//         for (int i = tId; i < sh_tile && (i +(tile_no*sh_tile)) < n_rows; i+=blockDim.x){
//             sh_r[t * sh_tile + i] = u[ (t * n_rows) + active_row[blockIdx.x*sh_tile+i]];
//         }
//     }    
//     __syncthreads();

//     for (int c = block_st + tId; c < block_lim; c+=blockDim.x){
//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         //int sh_row = passive_row[row] - blockIdx.x * sh_tile;
//         int sh_row = row - blockIdx.x * sh_tile;
//         // if( blockIdx.x < 3 && blockIdx.x <5 && sh_row< 2)
//         //     printf("GPU %d %d :: %d %d %d %d %f\n",c , blockIdx.x, row, active_row[row], passive_row[row], sh_row, sh_r[ sh_row] );

//         for (int t = 0; t < k; ++t)
//         {
//             //sm += u[row*k+t] * v[col*k+t];
//            // sm += sh_r[t * sh_tile + sh_row] * v[t*n_cols+col];
//             sm += sh_r[t * sh_tile + sh_row] * v[col*k+t];
//             //sm += u[t*n_rows+row] * v[t*n_cols+col];
//         }
//         val[c] = val[c] * sm; 

//     }  
// }

// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *active_row, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int gId = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh_c[96*32];
//     __shared__ float sh_r[96*32];
//     //__shared__ float sh_c[192*32];
//     int sh_tile = 96;
//     unsigned int c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;
//     if(c < tile_end && c< nnz){
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x){
//                 sh_c[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];
//             }
//         }
//     }           

//     __syncthreads();
//     int offset[96];

//     for (int c = tId + tile_st; c < tile_end && c< nnz; ){

//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_rows; i+=blockDim.x){
//                 sh_r[t * sh_tile + i] = u[ (t * n_rows) + (tile * sh_tile + active_row[i])];
//                 offset[i] = active_row[i] - i;
//             }
//         }    
//         __syncthreads();

//         float sm =0 ;
//         int row = row_ind[c];
//         row = offset[row];
//         int cc = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             float u_val = sh_r[(t * sh_tile) + (cc - tile * sh_tile)];
//             //u[t*n_rows+row];
//             sm += u_val * sh_c[(t * sh_tile) + (cc - tile * sh_tile)];
//         val[c] = val[c] * sm;  
//         c+=blockDim.x;      
//     }    
// }


// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];
//     int sh_tile = 180;
//     c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;

//     //if(tId < sh_tile && (tId +(tile*sh_tile)) < n_cols){
//     // if( c < tile_end && c < nnz ){ 
//     //     for (int t = 0; t < k; ++t){
//     //         for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//     //             sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];
//     //     }           
//     // }
//     // __syncthreads();

//         //if( c < nnz ){

//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int col = col_ind[c];
//         int tile  = col/sh_tile;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x)
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];

//         }    
//         __syncthreads();
//         float sm =0 ;
//         int row = row_ind[c];
//         //int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 

//     }

// }
// __global__ void comp_kernel_col_sorted_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];
//     int sh_tile = 180;
//     c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;

//     //if(tId < sh_tile && (tId +(tile*sh_tile)) < n_cols){
//     //if( c < tile_end && c< nnz )
//     // if(c < tId + tile_st + blockDim.x)
//     // { 
//     //     for (int t = 0; t < k; ++t){
//     //         for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//     //             sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];
//     //     }           
//     // }
//     // __syncthreads();

//         //if( c < nnz ){

//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int col = col_ind[c];
//         int tile  = col/sh_tile;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x)
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];

//         }    
//         __syncthreads();
//         float sm =0 ;
//         int row = row_ind[c];
//         //int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 

//     }

// }

// //CSR format
// __global__ void comp_kernel_CSR(int const* __restrict__ row_ptr, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//  float * p_ind, int n_rows, int k)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int laneId = tId & 31;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) >> 5;
//     extern __shared__ volatile  float SD[];

//     if( c < n_rows ){

//         // c = rowGroupPtr[c];  float g=0,h=0;
//         unsigned int row_p = row_ptr[c], nnz_row = row_ptr[c+1] - row_ptr[c];
//         for(long i=laneId; i < nnz_row; i += 32) {
//             float sm =0 ;
//             int row = c;
//             int col = col_ind[row_p + i];
//             for (int t = 0; t < k; ++t)
//                 sm += u[row*k+t] * v[col*k+t];
//             p_ind[row_p + i] = sm * val[row_p + i];


//         }
//         // __syncthreads();
//         if(c>500 && c< 505  && laneId == 0)
//             printf("from gpu %d %f\n",c,  p_ind[c] );
//         // float newvj=0;
//         // g += __shfl_down(g, 16);   
//         // g += __shfl_down(g, 8);
//         // g += __shfl_down(g, 4);   
//         // g += __shfl_down(g, 2);
//         // g += __shfl_down(g, 1);
//         // __syncthreads();

//     } 
// }


// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, 
//     const float * __restrict__ u, const float * __restrict__ v, float * p_ind, int nnz, int k){

//     unsigned int tId = threadIdx.x;
//     // unsigned int laneId = tId & 31;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;//>> 5;
//     // extern __shared__ volatile  float SD[];
//     // printf("I am in \n");
//     if( c < nnz ){

//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         printf("from gpu %d %f %d %f\n", c , p_ind[c], row, v[row*k+1] );

//         for (int t = 0; t < k; ++t)
//             sm += u[row*k+t] * v[col*k+t];
//         float s = .4;
//         p_ind[c] += val[c] * s;

//          // * val[c]; 

//         // // __syncthreads();
//         // if(c>5 && c< 14)
//         //     printf("from gpu %d %f\n",c,  p_ind[c] );      
//     } 



// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];

//     c = tId + tile_st;
//     int cc = col_ind[c];
//     int tile  = cc/180;

//         //if( c < nnz ){
//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int cc = col_ind[c];
//         int tile  = cc/180;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//                 sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];

//         }    
//         __syncthreads();
//     }

//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             sm += u[row*k+t] * sh[(t * 180) + (col - tile * 180)];
//         val[c] = val[c] * sm;
//     }
// }

//no blcok cyclic 
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;

//     int sh_tile = 180;
//     int cc = col_ind[c];
//     int tile  = cc/sh_tile;

//     int tile_st = tile_limIdx[tile];
//     //int tile_end = tile_limIdx[blockIdx.x+1];
//     int tile_end = tile_limIdx[tile+1];
//     // c = tId + tile_st;
//     if( c < tile_end+sh_tile ){
//         //if(c < tId + tile_st + blockDim.x)
//         __shared__ float sh[180*32];
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x){
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];
//             }
//         }   
//         __syncthreads();
//         if(c == 189520)printf("gpu 1 %d %d %d %d \n", c, cc, (0 * sh_tile) + (cc - tile * sh_tile), 0*n_cols+cc);
//         if( c < tile_end ){

//         //if(c % blockDim.x == 0 && blockIdx.x == 0) printf("blockcyclic %d %d \n", blockIdx.x, c);
//         //if(tId == 0 ) printf("gpu" ); 
//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         // if(sh[(0 * sh_tile) + (col - tile * sh_tile)] != v[0*n_cols+col] && col< 190){
//         //     printf("gpu 1 %d %d %f %f \n", c, col, sh[(0 * sh_tile) + (col - tile * sh_tile)], v[0*n_cols+col]);
//         //     printf("gpu 2 %d %d %f %f \n", c, col, sh[(1 * sh_tile) + (col - tile * sh_tile)], v[1*n_cols+col]);
//         // }

//         for (int t = 0; t < k; ++t){
//             //sm += u[row*k+t] * v[col*k+t];
//             //sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//             //if(c =55064677 && t ==6 ) printf("gpu %d %f %f \n", c, sh[(t * sh_tile) + (col - tile * sh_tile)], v[t*n_cols+col]);
//         }
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 
//     }
//     }

// }



