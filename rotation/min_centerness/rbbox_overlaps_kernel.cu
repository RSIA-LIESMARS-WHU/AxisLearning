
#include "rbbox_overlaps.hpp"
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cmath>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8; 


__device__ inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}

__device__ inline float area(float * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}
 
__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {



  if(num_of_inter > 0) {
    
    float center[2];
    
    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }
    
    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);
  
  if(area_abc * area_abd >= -1e-5) {
    return false;
  }
  
  area_cda = trangle_area(c, d, a); 
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);      
    
  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

__device__ inline bool inrect(float pt_x, float pt_y, const float * pts) {
  
  double ab[2];
  double ad[2];
  double ap[2];

  double abab;
  double abap;
  double adad;
  double adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];
  bool result = (abab - abap >=  -1) and (abap >= -1) and (adad - adap >= -1) and (adap >= -1);
  return result;
}

__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(inrect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(inrect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }   
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }
  

  return num_of_inter;
}

__device__ inline void convert_region(float *labelBoxPts, float *boundingRect, float *pivotPoints, float const * const region) {

  float angle = region[4];
  float a_cos = cos(angle/180.0*3.1415926535);
  float a_sin = -sin(angle/180.0*3.1415926535);// anti clock-wise
  
  float ctr_x = region[0];
  float ctr_y = region[1];
  float w = region[2];
  float h = region[3];

  float pts_x[4];
  float pts_y[4];

  pts_x[0] = - w / 2;
  pts_x[1] = - w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = w / 2;

  pts_y[0] = - h / 2;
  pts_y[1] = h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = - h / 2;

  // 中心轴2端点xyxy
  // float pivotPoints[4];
  if (w>h)
  {
    /*
    -----------
    |.23      |.01
    |    *    |
    -----------
    */
    pivotPoints[0] = a_cos * pts_x[2] + ctr_x;//0
    pivotPoints[1] = a_sin * pts_x[2] + ctr_y;//1
    pivotPoints[2] = a_cos * pts_x[1] + ctr_x;//2
    pivotPoints[3] = a_sin * pts_x[1] + ctr_y;//3
  }else{
  
  /*---.01-
    |     |
    |  *  |
    |     |
    ---.23--
    */
    pivotPoints[0] =  - a_sin * pts_y[0] + ctr_x;
    pivotPoints[1] =  a_cos * pts_y[0] + ctr_y;
    pivotPoints[2] = - a_sin * pts_y[1]  + ctr_x;
    pivotPoints[3] =  a_cos * pts_y[1]  + ctr_y;
  }
  // ltrd
  boundingRect[0] = 100000;
  boundingRect[1] = 100000;
  boundingRect[2] = 1;
  boundingRect[3] = 1;
  for(int i = 0;i < 4;i++) {
    labelBoxPts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    labelBoxPts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
    
    boundingRect[0] = min(labelBoxPts[2 * i], boundingRect[0]);
    boundingRect[1] = min(labelBoxPts[2 * i + 1], boundingRect[1]);
    boundingRect[2] = max(labelBoxPts[2 * i], boundingRect[2]);
    boundingRect[3] = max(labelBoxPts[2 * i + 1], boundingRect[3]);
  }

}



__device__ inline void get_BBox_pivotPts(float *box_along_vertic_length, float *boundingRect, float *pivotPoints, float const * const region) {


  pivotPoints[0] = (region[0] + region[2])/2.;//x1
  pivotPoints[1] = (region[1] + region[3])/2.;//y1
  pivotPoints[2] = (region[4] + region[6])/2.;//x2
  pivotPoints[3] = (region[5] + region[7])/2.;//y2
  
float len01_23 = sqrt(pow(region[2] - region[0], 2) + pow(region[3] - region[1], 2));
float len23_45 = sqrt(pow(region[4] - region[2], 2) + pow(region[5] - region[3], 2));
float len45_67 = sqrt(pow(region[6] - region[4], 2) + pow(region[7] - region[5], 2));
float len67_01 = sqrt(pow(region[0] - region[6], 2) + pow(region[1] - region[7], 2));
  box_along_vertic_length[0] = (len23_45 + len67_01)/2.;
  box_along_vertic_length[1] = (len01_23 + len45_67)/2.;
  // ltrd
  boundingRect[0] = 100000;
  boundingRect[1] = 100000;
  boundingRect[2] = 1;
  boundingRect[3] = 1;
  for(int i = 0;i < 4;i++) {    
    boundingRect[0] = min(region[2 * i], boundingRect[0]);
    boundingRect[1] = min(region[2 * i + 1], boundingRect[1]);
    boundingRect[2] = max(region[2 * i], boundingRect[2]);
    boundingRect[3] = max(region[2 * i + 1], boundingRect[3]);
  }

}

// __device__ inline float inter(float const * const region1, float const * const region2) {

//   // filter boxes
//   float pts1[8];
//   float pts2[8];
//   float int_pts[16];
//   int num_of_inter;

//   convert_region(pts1, region1);
//   convert_region(pts2, region2);

//   num_of_inter = inter_pts(pts1, pts2, int_pts);

//   reorder_pts(int_pts, num_of_inter);

//   return area(int_pts, num_of_inter);
  
  
// }

// __device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  
//   if((fabs(region1[0] - region2[0]) < 1e-5) && (fabs(region1[1] - region2[1]) < 1e-5) && (fabs(region1[2] - region2[2]) < 1e-5) && (fabs(region1[3] - region2[3]) < 1e-5) && (fabs(region1[4] - region2[4]) < 1e-5)) {
//     return 1.0;
//   }
  
//   float area1 = region1[2] * region1[3];
//   float area2 = region2[2] * region2[3];
//   float area_inter = inter(region1, region2);

//   float result = area_inter / (area1 + area2 - area_inter);

//   if(result < 0) {
//     result = 0.0;
//   }
//   return result;

  
// }
// __device__ inline void inv(const float *mat)
// {
//    __shared__ double MXI[2][2];
//  
//  int isx = threadIdx.x;
//  int isy = threadIdx.y;
//  double tmpIn;
//  double tmpInv;
//   //initialize E
//  if(isx == isy)
//  MXI[isy][isx] = 1;
//  else
//  MXI[isy][isx] = 0;
//  
//  for (int i = 0; i < 3; i++)
//  {
//     if (i == isy && isx < 3 && isy < 3)
//     {
//        //消除对角线上的元素（主元）为1
//        tmpIn = MX[i][i];
//        MX[i][isx] /= tmpIn;
//        MXI[i][isx] /= tmpIn;
//     }
//    __syncthreads();
//    if (i != isy && isx < 3 && isy < 3)
//    {
//      //将主元所在列的元素化为0 所在行的元素同时变化
//       tmpInv = MX[isy][i];
//       MX[isy][isx] -= tmpInv * MX[i][isx];
//       MXI[isy][isx] -= tmpInv * MXI[i][isx];
//    }
//    __syncthreads();
//  }
// }

__device__ inline float gauss_weight(const float c_x, const float c_y, const float sigma, const int grid_pt_x, const int grid_pt_y){
  return exp(-(pow(grid_pt_x - c_x, 2) + pow(grid_pt_y - c_y,  2))  / (2 * pow(sigma, 2)) );
}

__device__ inline float fcos_weight(const int pt_x, const int pt_y, 
                                    const float a_x, const float a_y, const float b_x, const float b_y,
                                    const float long_w, const float short_h){
  float a_pt[2], a_b[2];
  a_pt[0] = pt_x - a_x;
  a_pt[1] = pt_y - a_y;
  a_b[0] = b_x - a_x;
  a_b[1] = b_y - a_y;
  float pt_proj_ab_length = (a_pt[0]*a_b[0] + a_pt[1]*a_b[1])/sqrt(a_b[0]*a_b[0]+a_b[1]*a_b[1]);
  float pt_2_ab_length = sqrt(a_pt[0]*a_pt[0]+a_pt[1]*a_pt[1] - pt_proj_ab_length*pt_proj_ab_length);
  // ori centerness
  float temp = min(pt_proj_ab_length, long_w - pt_proj_ab_length)/max(pt_proj_ab_length, long_w - pt_proj_ab_length)*\
               min(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length)/max(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length);
  
  // min center
  // float temp = min(min(pt_proj_ab_length, long_w - pt_proj_ab_length)*2./long_w ,
  //                  min(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length)*2. /short_h);

  // if(temp<0) 
  // printf("%f %f %f\n", temp, pt_proj_ab_length, pt_2_ab_length);
  return sqrt(max(0., temp));//max(0., temp);//sqrt(max(0., temp));
}

// __device__ void PRINT(int d)
// {
//   printf("%d", d);
// }

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_grid_points, float* gridpts2targets_dev) {

// -------> target/threadsPerBlock
// |_|_|_|_|_|_|_ 每个格子都是一个block
// |_|_|_|_|_|_|_
// |_|_|_|_|_|_|_
// |
// anchor/threadsPerBlock
  const int col_start = blockIdx.y;//anchor |  ;//grid_pts
  const int row_start = blockIdx.x;//target start-->  //target_box


  const int row_size =
        min(N - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(K - col_start * threadsPerBlock, threadsPerBlock);


  // __shared__ float block_boxes[threadsPerBlock * 5];
  __shared__ float block_grid_points[threadsPerBlock * 2];
  if (threadIdx.x < col_size) {
    block_grid_points[threadIdx.x * 2 + 0] =
        dev_grid_points[(threadsPerBlock * col_start + threadIdx.x) * 2 + 0];
    block_grid_points[threadIdx.x * 2 + 1] =
        dev_grid_points[(threadsPerBlock * col_start + threadIdx.x) * 2 + 1];
  }

  // dev_boxes 在外面 block_boxes 是共享显存
  // if (threadIdx.x < row_size) {
  //   block_boxes[threadIdx.x * 5 + 0] =
  //       dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 0];//c_x
  //   block_boxes[threadIdx.x * 5 + 1] =
  //       dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 1];//c_y
  //   block_boxes[threadIdx.x * 5 + 2] =
  //       dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 2];//w
  //   block_boxes[threadIdx.x * 5 + 3] =
  //       dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 3];//h
  //   block_boxes[threadIdx.x * 5 + 4] =
  //       dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 4];//theta
  // }

  __syncthreads();


  if (threadIdx.x < row_size) {
    
const float * box =  dev_boxes + (threadsPerBlock * row_start + threadIdx.x) * 8;
// 计算一个target的中心轴的两端点
  // float labelBoxPts[8];//4个顶点坐标
  float boundingRect[4];//ltrd for fast filter
  float pivotPoints[4];//中心轴端点坐标xyxy
  float box_along_vertic_length[2];//通过四个点获取wh

  // int num_of_inter;

  // xywha
  // convert_region(labelBoxPts, boundingRect, pivotPoints, block_boxes + threadIdx.x * 5);
  // 4xy
  get_BBox_pivotPts(box_along_vertic_length, boundingRect, pivotPoints, box);


  int box_id = row_start*threadsPerBlock + threadIdx.x;
  // printf("%d %d \n", threadIdx.x, box_id);
  // float box_cx = *(block_boxes + threadIdx.x * 5 + 0);
  // float box_cy = *(block_boxes + threadIdx.x * 5 + 1);
  // float box_w = *(box + 2);
  // float box_h = *(box + 3);
  float box_area = box_along_vertic_length[0]*box_along_vertic_length[1];

// if (row_size==1)
//     printf("%d %d %d\n", threadIdx.x, row_size, box_id);

  for(int i = 0;i < col_size; i++) {
     
      // int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i;
      int offset = (col_start*threadsPerBlock + i) * 8;
      // dev_overlaps[offset] = devRotateIoU(block_boxes + threadIdx.x * 5, block_query_boxes + i * 5);
      // 过滤
      float grid_pt_x = *(block_grid_points + 2*i);
      float grid_pt_y = *(block_grid_points + 2*i +1);
      
      // gridpts2targets_dev[offset] = -1;
      // 落在外接矩形外面 或者 落在外接矩形内部但是没有落在目标框内
      if ((grid_pt_y <= boundingRect[1]) || (grid_pt_y >= boundingRect[3]) || 
          (grid_pt_x <= boundingRect[0]) || (grid_pt_x >= boundingRect[2]) ||
          ! inrect(grid_pt_x, grid_pt_y, box) ){
            
           continue;
      }else if (gridpts2targets_dev[offset] < 0.5 && gridpts2targets_dev[offset+1]<0.5 || 
                box_area < gridpts2targets_dev[offset+1]){
          // printf("%f %f %f", grid_pt_x, grid_pt_y, gridpts2targets_dev[offset]);
          //没有加冗余信息 中心轴端点坐标xyxy w 缺少类别信息
          // 有正有fu
          
          // float ori_value = gridpts2targets_dev[offset];
          gridpts2targets_dev[offset] = box_id;//target_idx
            // printf("%f %d %f\n", ori_value, box_id, gridpts2targets_dev[offset]);
          gridpts2targets_dev[offset+1] = box_area;//target area
          gridpts2targets_dev[offset+2] = block_grid_points[2 * i] - pivotPoints[0];//detax1
          gridpts2targets_dev[offset+3] = block_grid_points[2 * i + 1] - pivotPoints[1];//detay1
          gridpts2targets_dev[offset+4] = block_grid_points[2 * i] - pivotPoints[2];//detax2
          gridpts2targets_dev[offset+5] = block_grid_points[2 * i + 1] - pivotPoints[3];//detay2
          
          gridpts2targets_dev[offset+6] = box_along_vertic_length[1];//最短边
          
          // const int pt_x, const int pt_y, 
          //                           const float a_x, const float a_y, const float b_x, const float b_y,
          //                           const float long_w, const float short_h
          // float weight = gauss_weight(box_cx, box_cy, 0.15*min(box_h, box_w), grid_pt_x, grid_pt_y);// 权重
          float weight =  fcos_weight(grid_pt_x, grid_pt_y, \
                                                      pivotPoints[0], pivotPoints[1], pivotPoints[2], pivotPoints[3],\
                                                      box_along_vertic_length[0], box_along_vertic_length[1]);// 权重
          // printf("%f\n", weight);
          // if (weight<0.5)
          //    weight = 0;
             
          gridpts2targets_dev[offset+7] = weight;
          
          // printf("%f %f %f %f %f %f %f %f\n", gridpts2targets_dev[offset], gridpts2targets_dev[offset+1], gridpts2targets_dev[offset+2], gridpts2targets_dev[offset+3],
          //                                     gridpts2targets_dev[offset+4], gridpts2targets_dev[offset+5], gridpts2targets_dev[offset+6], gridpts2targets_dev[offset+7]);

      }      
  }//for

}
}


void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}


void _overlaps(float* targets2gridpts, const float* label_boxes,const float* grid_points, int n, int k, int device_id) {
// n target
// k grid_points
  _set_device(device_id);

  float* gridpts2targets_dev = NULL;
  // xywha & cxcy w h A
  float* boxes_dev = NULL;
  float* grid_points_dev = NULL;

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        n * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        label_boxes,
                        n * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));



  CUDA_CHECK(cudaMalloc(&grid_points_dev,
                        k * 2 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(grid_points_dev,
                        grid_points,
                        k * 2 * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  // 每个坐标点适配信息 适配target_idx target_area x1 y1 x2 y2 h centerness
  CUDA_CHECK(cudaMalloc(&gridpts2targets_dev,
                        k * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(gridpts2targets_dev,
                        targets2gridpts,
                        k * 8 * sizeof(float),
                        cudaMemcpyHostToDevice));


  if (true){}

  
  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(k, threadsPerBlock));
		
  dim3 threads(threadsPerBlock);

  overlaps_kernel<<<blocks, threads>>>(n, k,
                                    boxes_dev,
                                    grid_points_dev,
                                    gridpts2targets_dev);  


  
  CUDA_CHECK(cudaMemcpy(targets2gridpts,
                        gridpts2targets_dev,
                        k * 8 * sizeof(float),
                        cudaMemcpyDeviceToHost));



  

  

  CUDA_CHECK(cudaFree(gridpts2targets_dev));

  CUDA_CHECK(cudaFree(boxes_dev));

  CUDA_CHECK(cudaFree(grid_points_dev));

}
