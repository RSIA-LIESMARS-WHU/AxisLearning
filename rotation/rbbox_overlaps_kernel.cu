
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
__constant__ double PI = 3.1415926535;

// chacheng
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
// 顺时针顺序排列
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
      v[0] = v[0] / d;//cos
      v[1] = v[1] / d;//sin
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
  bool result1 = (abab - abap >=  -1) and (abap >= -1) and (adad - adap >= -1) and (adap >= -1);
  return result1;
  // ab[0] = pts[2] - pts[4];
  // ab[1] = pts[3] - pts[5];
  // ad[0] = pts[6] - pts[4];
  // ad[1] = pts[7] - pts[5];
  // ap[0] = pt_x - pts[4];
  // ap[1] = pt_y - pts[5];
  // abab = ab[0] * ab[0] + ab[1] * ab[1];
  // abap = ab[0] * ap[0] + ab[1] * ap[1];
  // adad = ad[0] * ad[0] + ad[1] * ad[1];
  // adap = ad[0] * ap[0] + ad[1] * ap[1];
  // bool result2 = (abab - abap >=  -1) and (abap >= -1) and (adad - adap >= -1) and (adap >= -1);
  // return result1 or result2;
}

__device__ inline bool inPolygon(float pt_x, float pt_y, const float * pts) {
  
  float abap, bcbp, cdcp, dadp;

  // ab[0] = pts[2] - pts[0];
  // ab[1] = pts[3] - pts[1];
  // ap[0] = pt_x - pts[0];
  // ap[1] = pt_y - pts[1];

  //   ab[0] = pts[4] - pts[2];
  // ab[1] = pts[5] - pts[3];
  // ap[0] = pt_x - pts[2];
  // ap[1] = pt_y - pts[3];

  //   ab[0] = pts[6] - pts[4];
  // ab[1] = pts[7] - pts[5];
  // ap[0] = pt_x - pts[4];
  // ap[1] = pt_y - pts[5];

  //   ab[0] = pts[0] - pts[6];
  // ab[1] = pts[1] - pts[7];
  // ap[0] = pt_x - pts[6];
  // ap[1] = pt_y - pts[7];

  // abab = ab[0] * ab[0] + ab[1] * ab[1];
  // 叉乘
  abap = (pts[2] - pts[0])* (pt_y - pts[1]) - (pts[3] - pts[1]) * (pt_x - pts[0]);
  bcbp = (pts[4] - pts[2])* (pt_y - pts[3]) - (pts[5] - pts[3])*(pt_x - pts[2]) ;
  cdcp = (pts[6] - pts[4])* (pt_y - pts[5]) - (pts[7] - pts[5]) *(pt_x - pts[4]) ;
  dadp = (pts[0] - pts[6])* (pt_y - pts[7]) - (pts[1] - pts[7])*(pt_x - pts[6]);
  // if( (abap>0 && bcbp>0 && cdcp>0 && dadp>0) || (abap<0 && bcbp<0 && cdcp<0 && dadp<0))

// y轴向下
  if(abap>0 && bcbp>0 && cdcp>0 && dadp>0 || (abap<0 && bcbp<0 && cdcp<0 && dadp<0)){
      // printf("%f,%f  %f,%f  %f,%f %f,%f　 %f－%f \n", pts[0], pts[1], pts[2], pts[3], 
  // pts[4], pts[5], pts[6], pts[7], pt_x,pt_y);
  return true;
  }
    
  return false;
}


int cross(float ax, float ay, float bx, float by, float cx, float cy)//计算叉积
{
    return (bx-ax)*(cy-ay)-(cx-ax)*(by-ay);
}

// bool cmp_angle(float ax,float ay, float bx, float by)//极角排序另一种方法，速度快
// {
//     if(atan2(ay-yy, ax-xx)!=atan2(by-yy,bx-xx))
//         return (atan2(ay-yy, ax-xx))<(atan2(by-yy,bx-xx));
//     return ax<bx;
// }

__device__ inline void bubble_sort(int data_num, float*angle, const float*coord, int* index)
{
  for(int i=0; i<data_num-1; i++)
    index[i]=i;
  
  // 从小到大
  bool flag = true;//发生排序
  for(int i=0; i<data_num-1 && flag; i++){
      flag = false;
      // max index
      for(int j = data_num-1; j>i; j--){ //和(j=0;j<data_num-i-1;j++)相同
      // 极角从大到小排序　若相等则按水平距离排序
       if( angle[j] > angle[j-1] || (angle[j] == angle[j-1] && coord[j*2] > coord[j*2 -2])){
              flag = true;
              float tmp = angle[j];
              angle[j] = angle[j-1];
              angle[j-1] = tmp;
              // 交换位置index
              int index_temp = index[j-1];
              index[j-1] = index[j];
              index[j] = index_temp; 
          }
      }
  }
}


__device__ inline float convex_hull(const int num_of_ourer, const float * out_pts) {
int max_y = -1000000;
int max_id = -1;

//得到最大y坐标点id 若y值相同最左边的点
for(int i = 0;i < num_of_ourer;i++)
{
  if (out_pts[i*2+1] >= max_y)
  {
    if (out_pts[i*2+1] == max_y && out_pts[i*2] > out_pts[max_id*2])
      continue;
    max_y = out_pts[i*2+1];
    max_id = i;
  }
}
float stack[50];//凸包中所有的点xy max:8*2
stack[0] = out_pts[max_id*2];
stack[1] = out_pts[max_id*2+1];

float angle[10];
int count=0;
for(int i=0;i<num_of_ourer && i!= max_id; i++)
{
  angle[count++] = atan2(out_pts[2*i+1]-stack[1], out_pts[2*i]-stack[0]);
}

// 按极角排序
int index[10];
bubble_sort(num_of_ourer-1, angle, out_pts, index);

// 加入最大极角点
stack[2] = out_pts[index[0]*2];
stack[3] = out_pts[index[0]*2+1];
int top=1;

for(int i=2; i<num_of_ourer; i++)
{
  float pt0x = stack[2*top-2];
  float pt0y = stack[2*top-1];
  float pt1x = stack[2*top];
  float pt1y = stack[2*top+1]; 
  float pt2x = out_pts[index[i]*2]; 
  float pt2y = out_pts[index[i]*2+1];
  while(i>=1 && top>0 && cross(pt0x, pt0y, pt1x, pt1y, pt2x, pt2y)<0)
    top--;
  top++;
  stack[top*2] = pt2x;
  stack[top*2+1] = pt2y;
}
return area(stack, top+1);

}



__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts, float* out_pts, int* num_of_outer) {

  int num_of_inter = 0;
  for(int i = 0;i < 4;i++) {
    if(inrect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }else{
      out_pts[*num_of_outer * 2] = pts1[2 * i];
      out_pts[*num_of_outer * 2 + 1] = pts1[2 * i + 1];
      (*num_of_outer)++;
    }
     if(inrect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }else{
      out_pts[*num_of_outer * 2] = pts1[2 * i];
      out_pts[*num_of_outer * 2 + 1] = pts1[2 * i + 1];
      (*num_of_outer)++;
    }
  }

  float temp_pts[2];
  // 交点
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
// 逆时针
  float angle = region[4];
  float a_cos = cos(angle/180.0*PI);
  float a_sin = -sin(angle/180.0*PI);// anti clock-wise
  
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

__device__ inline void convert_xyxya_4xy(const float *predictPts, float  *region) {
  // predictPts x1y1 x3y3 w
  //         *3
  //   *2  *
  //     *    
  //   *   *4
  // *1   
  // 逆时针
  float angle = atan2(predictPts[1]-predictPts[3], predictPts[0]-predictPts[2]);
  float w_a = 0;
  if(angle+PI/2. < PI )
    w_a = angle+PI/2.;
  else
    w_a = angle+PI/2.-2*PI;
  
  float cy = (predictPts[3] + predictPts[1])/2.;
  float cx = (predictPts[2] + predictPts[0])/2.;
  float p2x = cx + predictPts[4]/2.*cos(w_a);
  float p2y = cy + predictPts[4]/2.*sin(w_a);
  float p4x = 2*cx - p2x;
  float p4y = 2*cy - p2y;
  region[0] = predictPts[0] + p2x - cx;
  region[1] = predictPts[1] + p2y - cy;
  region[2] = predictPts[0] + p4x - cx;
  region[3] = predictPts[1] + p4y - cy;
  region[4] = predictPts[2] + p4x - cx;
  region[5] = predictPts[3] + p4y - cy;
  region[6] = predictPts[2] + p2x - cx;
  region[7] = predictPts[3] + p2y - cy;
}




__device__ inline void get_BBox_pivotPts(float *box_along_vertic_length, float *boundingRect, float *diamondRegion, float *pivotPoints, float const * const region) {

  pivotPoints[0] = (region[0] + region[2])/2.;//x1
  pivotPoints[1] = (region[1] + region[3])/2.;//y1
  pivotPoints[2] = (region[4] + region[6])/2.;//x2
  pivotPoints[3] = (region[5] + region[7])/2.;//y2
  // pivotPoints[0] = (region[0] + region[2])/2.;//x1
  // pivotPoints[1] = (region[1] + region[3])/2.;//y1
  // pivotPoints[2] = (region[4] + region[6])/2.;//x2
  // pivotPoints[3] = (region[5] + region[7])/2.;//y2
  diamondRegion[0] = (region[0] + region[2])/2.;
  diamondRegion[1] = (region[1] + region[3])/2.;
  diamondRegion[2] = (region[2] + region[4])/2.;
  diamondRegion[3] = (region[3] + region[5])/2.;
  diamondRegion[4] = (region[4] + region[6])/2.;
  diamondRegion[5] = (region[5] + region[7])/2.;
  diamondRegion[6] = (region[6] + region[0])/2.;
  diamondRegion[7] = (region[7] + region[1])/2.;

  //   diamondRegion[0] = ((region[0] + region[2])/2.- cx)*3/4.+cx;
  // diamondRegion[1] = ((region[1] + region[3])/2.- cy)*4/5.+cy;
  // diamondRegion[2] = ((region[2] + region[4])/2.- cx)*3/4.+cx;
  // diamondRegion[3] = ((region[3] + region[5])/2.- cy)*4/5.+cy;
  // diamondRegion[4] = ((region[4] + region[6])/2.- cx)*3/4.+cx;
  // diamondRegion[5] = ((region[5] + region[7])/2.- cy)*4/5.+cy;
  // diamondRegion[6] = ((region[6] + region[0])/2.- cx)*3/4.+cx;
  // diamondRegion[7] = ((region[7] + region[1])/2.- cy)*4/5.+cy;
  
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
  float pt_proj_ab = (a_pt[0]*a_b[0] + a_pt[1]*a_b[1])/sqrt(a_b[0]*a_b[0]+a_b[1]*a_b[1]);
  float pt_2_ab = sqrt(a_pt[0]*a_pt[0]+a_pt[1]*a_pt[1] - pt_proj_ab*pt_proj_ab);
  float temp = 0;
  // if (long_w < short_h)
  // {
  //     // ori centerness
  //      temp = min(pt_proj_ab, long_w - pt_proj_ab)*2/long_w*\
  //                  min(short_h/2. - pt_2_ab, short_h/2. + pt_2_ab)/max(short_h/2. - pt_2_ab, short_h/2. + pt_2_ab);
  // }else{
  //     temp = min(pt_proj_ab, long_w - pt_proj_ab)/max(pt_proj_ab, long_w - pt_proj_ab)*\
  //                  min(short_h/2. - pt_2_ab, short_h/2. + pt_2_ab)*2/short_h;
  // }

// ori centerness
    temp = min(pt_proj_ab, long_w - pt_proj_ab)/max(pt_proj_ab, long_w - pt_proj_ab)*\
                   min(short_h/2. - pt_2_ab, short_h/2. + pt_2_ab)/max(short_h/2. - pt_2_ab, short_h/2. + pt_2_ab);
  return sqrt(max(0., temp));
  
  // min center
//   float temp = min(min(pt_proj_ab_length, long_w - pt_proj_ab_length)*2./long_w ,
//                    min(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length)*2. /short_h);
// return max(0., temp);

  // float temp = min(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length)/max(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length);
  // // if(temp<0) 
  // // printf("%f %f %f\n", temp, pt_proj_ab_length, pt_2_ab_length);
  // return max(0., temp);//max(0., temp);//sqrt(max(0., temp));

  // add ratio
}

// __device__ void PRINT(int d)
// {
//   printf("%d", d);
// }



__global__ void IOU_kernel(const int N, float* targets_value_dev,
                               const float * predict_value_dev, float* IOU_dev) {

// -------> target/threadsPerBlock
// |_|_|_|_|_|_|_ 每个格子都是一个block

// anchor/threadsPerBlock
  const int row_start = blockIdx.x;//target start-->  //target_box

float * target =  targets_value_dev + (threadsPerBlock * row_start + threadIdx.x) * 8;
const float * predict =  predict_value_dev + (threadsPerBlock * row_start + threadIdx.x) * 5;
float target_cx = (target[0]+target[2]+target[4]+target[6])/4.;
float target_cy = (target[1]+target[3]+target[5]+target[7])/4.;
float predict_cx = (predict[0]+predict[2])/2.;
float predict_cy = (predict[1]+predict[3])/2.;

// filter boxes
float predictPts[8];
float int_pts[16];
float out_pts[16];
int num_of_inter;
convert_xyxya_4xy(predict, predictPts);

// convert_region(pts1, region1);
// convert_region(pts2, region2);
// conver x1y1 x2y2 w -> 4xy
int num_of_outer=0;
num_of_inter = inter_pts(target, predictPts, int_pts, out_pts, &num_of_outer);
reorder_pts(int_pts, num_of_inter);
float interArea = area(int_pts, num_of_inter);
// float generalArea = convex_hull(out_pts, num_of_outer);
float targetArea = area(target, 4);
float predictArea = area(predictPts, 4);
IOU_dev[threadsPerBlock * row_start + threadIdx.x] = interArea/(targetArea+predictArea-interArea);

// float generalIOU = IOU - (generalArea - )
// IOULoss_dev[threadsPerBlock * row_start + threadIdx.x] = 1-( - )

}

//  [-1, 64],
  // [32, 128],
  // [64, 256],
  // [128, 512],
  // [256, INF],
  // __constant__ float valid_range[10] = {-1,64,54,128,118,256,246,512,502,100000};  
__constant__ float valid_range[10] = {-1,64,32,128,64,256,128,512,256,100000};  

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
    // block_grid_points[threadIdx.x * 3 + 2] =
    //     dev_grid_points[(threadsPerBlock * col_start + threadIdx.x) * 3 + 2];
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
  // float labelBoxPts[8];//4个顶点坐标
  float boundingRect[4];//ltrd for fast filter
  float diamondRegion[8];//中心轴端点坐标xyxy
  float box_along_vertic_length[2];//通过四个点获取wh

  // int num_of_inter;

  // xywha
  // convert_region(labelBoxPts, boundingRect, pivotPoints, block_boxes + threadIdx.x * 5);
  // 4xy
   float pivotPoints[4];//中心轴端点坐标xyxy
  get_BBox_pivotPts(box_along_vertic_length, boundingRect, diamondRegion, pivotPoints, box);


  int box_id = row_start*threadsPerBlock + threadIdx.x;

  // float box_w = *(box + 2);
  // float box_h = *(box + 3);
  float box_area = box_along_vertic_length[0]*box_along_vertic_length[1];
  float sidelength = sqrt(box_area);
  double aspect_ratio = max(box_along_vertic_length[0]*1. / box_along_vertic_length[1], 
                           box_along_vertic_length[1]*1. / box_along_vertic_length[0]);

  aspect_ratio = pow(aspect_ratio, 1/3.);

  for(int i = 0;i < col_size; i++) {
     
      // int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i;
      int offset = (col_start*threadsPerBlock + i) * 8;
      // dev_overlaps[offset] = devRotateIoU(block_boxes + threadIdx.x * 5, block_query_boxes + i * 5);
      // 过滤
      float grid_pt_x = *(block_grid_points + 2*i);
      float grid_pt_y = *(block_grid_points + 2*i +1);
      int grid_pt_level = *(block_grid_points + 2*i +2);

      // printf("hello\n");
          // printf("%d %f %f %f\n", grid_pt_level, box_along_vertic_length[0],box_along_vertic_length[1], valid_range[grid_pt_level*2]);


      // gridpts2targets_dev[offset] = -1;
      // 落在外接矩形外面 或者 落在外接矩形内部但是没有落在目标框内
      // diamondRegion
      if ( (grid_pt_y <= boundingRect[1]) || (grid_pt_y >= boundingRect[3]) || 
          (grid_pt_x <= boundingRect[0]) || (grid_pt_x >= boundingRect[2]) ||
          ! inrect(grid_pt_x, grid_pt_y, box) ){
            
           continue;
          
          // ((sidelength > valid_range[grid_pt_level*2] && sidelength <  valid_range[grid_pt_level*2+1]) && 
                // (gridpts2targets_dev[offset] < 0.5 && gridpts2targets_dev[offset+1]<0.5 || 
                // box_area < gridpts2targets_dev[offset+1]))
                // (sidelength > valid_range[grid_pt_level*2] && sidelength <  valid_range[grid_pt_level*2+1]) && 
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
          float centerness =  fcos_weight(grid_pt_x, grid_pt_y, \
                                      diamondRegion[0], diamondRegion[1], diamondRegion[4], diamondRegion[5],\
                                      box_along_vertic_length[0], box_along_vertic_length[1]);// 权重
          // printf("%f\n", weight);
          // if (weight<0.5)
          //    weight = 0;
             
          gridpts2targets_dev[offset+7] = min(aspect_ratio*centerness, 1.);//centerness;//min(aspect_ratio*centerness, 1.);// //min(aspect_ratio*centerness, 1.);//min(aspect_ratio*centerness, 1.);//weight for loss


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

// 4xy level
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



void _IOU(const float* targets_value, const float* predict_value, float* IOU, int n, int device_id) {
// n target
// k grid_points
  _set_device(device_id);
  float* IOU_dev = NULL;
  float* targets_value_dev = NULL;
  // xywha & cxcy w h A
  float* predict_value_dev = NULL;

  // xywha
  CUDA_CHECK(cudaMalloc(&targets_value_dev,
                        n * 5 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(targets_value_dev,
                        targets_value,
                        n * 5 * sizeof(float),
                        cudaMemcpyHostToDevice));
  // x1y1x2y2w
  CUDA_CHECK(cudaMalloc(&predict_value_dev,
                        n * 5 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(predict_value_dev,
                        predict_value,
                        n * 5 * sizeof(float),
                        cudaMemcpyHostToDevice));

  
  dim3 blocks(DIVUP(n, threadsPerBlock));
		
  dim3 threads(threadsPerBlock);

  IOU_kernel<<<blocks, threads>>>(n,targets_value_dev,
                                          predict_value_dev,
                                          IOU_dev);  


  
  CUDA_CHECK(cudaMemcpy(IOU,
                        IOU_dev,
                        n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(targets_value_dev));

  CUDA_CHECK(cudaFree(predict_value_dev));

}



// // 为每一个box筛选出topk个预测最准确的位置
// void _LOSS2weight(const float* loss_with_boxid, const float* box_loss_weight, int n, float topk) {
// // n target
// // k grid_points
//   _set_device(device_id);
//   float* IOU_dev = NULL;
//   float* targets_value_dev = NULL;
//   // xywha & cxcy w h A
//   float* predict_value_dev = NULL;

//   // xywha
//   CUDA_CHECK(cudaMalloc(&targets_value_dev,
//                         n * 5 * sizeof(float)));
//   CUDA_CHECK(cudaMemcpy(targets_value_dev,
//                         targets_value,
//                         n * 5 * sizeof(float),
//                         cudaMemcpyHostToDevice));
//   // x1y1x2y2w
//   CUDA_CHECK(cudaMalloc(&predict_value_dev,
//                         n * 5 * sizeof(float)));
//   CUDA_CHECK(cudaMemcpy(predict_value_dev,
//                         predict_value,
//                         n * 5 * sizeof(float),
//                         cudaMemcpyHostToDevice));

  
//   dim3 blocks(DIVUP(n, threadsPerBlock));
		
//   dim3 threads(threadsPerBlock);

//   IOU_kernel<<<blocks, threads>>>(n,targets_value_dev,
//                                           predict_value_dev,
//                                           IOU_dev);  


  
//   CUDA_CHECK(cudaMemcpy(IOU,
//                         IOU_dev,
//                         n * sizeof(float),
//                         cudaMemcpyDeviceToHost));

//   CUDA_CHECK(cudaFree(targets_value_dev));

//   CUDA_CHECK(cudaFree(predict_value_dev));

// }