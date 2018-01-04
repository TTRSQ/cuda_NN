
#include <iostream>
#include <cstdio>

#define BLOCK_SIZE 16
#define SIZE_RATE 10

struct matrix{
  int height;
  int width;
  float *elements;
};

void matrixSizeInit(matrix M, int h, int w){
  M.height = h;
  M.width = w;

  int size = h*w;

  delete [] M.elements;
  M.elements = new float[size];
  for(int i = 0; i < size; i++) M.elements[i] = 0.0;
}

void sizeInitFromMatrix(matrix M, matrix from){
  M.height = from.height;
  M.width = from.width;

  int size = from.height*from.width;

  delete [] M.elements;
  M.elements = new float[size];
  for(int i = 0; i < size; i++) M.elements[i] = 0.0;
}

void printMatrix(matrix M){
  for(int i = 0; i < M.height; i++){
    for(int j = 0; j < M.width; j++){
      if(j != 0) std::cout << " ";
      std::cout << M.elements[i*M.width+j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

__global__ void matrixAdd(matrix M, matrix add){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] += add.elements[row*add.width+col];
  }
}

__global__ void matrixMul(matrix A, matrix B, matrix C){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < C.height && col < C.width){
    float x = 0.0f;
    for (int i = 0; i < A.width; i++) {
      x += A.elements[row*A.width+i]*B.elements[i*B.width+col];
    }

    C.elements[row*C.width+col] = x;
  }
}

static void matrixMul_gpu(matrix& d_m_in, matrix& d_m_ac){
  //デバイスに演算結果の領域を確保
  matrix d_ans;
  d_ans.width = d_m_ac.width; d_ans.height = d_m_in.height;

  int size;
  //デバイスにメモリ確保
  size = d_ans.width*d_ans.height*sizeof(float);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_ans.width-1+blk.x)/blk.x, (d_ans.height-1+blk.y)/blk.y);

  matrixMul<<<gld, blk>>>(d_m_in, d_m_ac, d_ans);

  matrix A;
  A.height = 2;
  A.width = 2;
  A.elements = new float[A.height*A.width];
  cudaMemcpy(A.elements, d_m_ac.elements, size, cudaMemcpyDeviceToHost);
  std::cout << "matrix:actcpy =" << std::endl;
  printMatrix(A);

  //不要になった入力のメモリの開放
  cudaFree(d_m_in.elements);

  //演算結果を引き継ぐ
  d_m_in = d_ans;
}

__global__ void matrixAddBias(matrix M, matrix bias){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] += bias.elements[col];
  }
}

__global__ void matrixRelu(matrix M){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] = (M.elements[row*M.width+col] < 0)? 0: M.elements[row*M.width+col];
  }
}

void randomInit(matrix m, int maxVal){
  for(int i = 0; i < m.height*m.width; i++) m.elements[i] = maxVal*(float(rand())/RAND_MAX) - maxVal/2.0;
}

__gloval__ void name(matrix a) {

}

void checkFunction(){
  matrix A, B;
  A.height = A.width = 2;
  B.height = B.width = 2;

  A.elements = new float[A.width*A.height];
  B.elements = new float[B.width*B.height];

  randomInit(A, 10);
  randomInit(B, 10);

  //演算前確認
  std::cout << "matrix:A =" << std::endl;
  printMatrix(A);
  std::cout << "matrix:B =" << std::endl;
  printMatrix(B);

  matrix dA, dB;
  dA.width = A.width; dA.height = A.height;
  dB.width = B.width; dB.height = B.height;

  int a;
  &a

  int *a;
  *a = 1;
  a = 1が入ってるアドレス
  int b = 2;
  a = &b;
  &a//=2

  int size = dA.width*dA.height*sizeof(float);
  cudaMalloc((void**)&dA.elements, size);
  cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

  cudaMemcpy(B.elements, dA.elements, size, cudaMemcpyDeviceToHost);

  std::cout << "matrix:B =" << std::endl;
  printMatrix(B);

  // size = dB.width*dB.height*sizeof(float);
  // cudaMalloc((void**)&dB.elements, size);
  // cudaMemcpy(dB.elements, B.elements, size, cudaMemcpyHostToDevice);
  //
  // func(dA, dB);
  //
  // cudaMemcpy(A.elements, dA.elements, size, cudaMemcpyDeviceToHost);
  // std::cout << "matrix:ans =" << std::endl;
  // printMatrix(A);
  //
  // // ホストメモリ解放
  // delete [] A.elements;
  // delete [] B.elements;
}

int main(){
  checkFunction();
  return 0;
}
