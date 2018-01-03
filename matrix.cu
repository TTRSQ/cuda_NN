
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

__global__ void matrixAdd(matrix A, matrix B, matrix C){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < C.height && col < C.width){
    C.elements[row*C.width+col] = A.elements[row*A.width+col] + B.elements[row*B.width+col];
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

__global__ void matrixBias(matrix A, matrix B, matrix C){
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

__global__ void matrixRelu(matrix A, matrix B, matrix C){
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



matrix matrixMul_gpu(matrix A, matrix B){
  matrix C;
  C.height = A.height;
  C.width = B.width;
  C.elements = new float[C.height*C.width];
  //デバイス要変数の用意
  matrix dA, dB, dC;
  dA.width = A.width; dA.height = A.height;
  dB.width = B.width; dB.height = B.height;
  dC.width = C.width; dC.height = C.height;

  int size;
  // デバイスメモリの確保とホストからの転送
  size = dA.width*dA.height*sizeof(float);
  cudaMalloc((void**)&dA.elements, size);
  cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

  size = dB.width*dB.height*sizeof(float);
  cudaMalloc((void**)&dB.elements, size);
  cudaMemcpy(dB.elements, B.elements, size, cudaMemcpyHostToDevice);

  //Cは計算前なのでコピー不要
  size = dC.width*dC.height*sizeof(float);
  cudaMalloc((void**)&dC.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((C.width-1+blk.x)/blk.x, (C.height-1+blk.y)/blk.y);

  matrixMul<<<gld, blk>>>(dA, dB, dC);

  //sizeはそのまま
  cudaMemcpy(C.elements, dC.elements, size, cudaMemcpyDeviceToHost);

  //デバイスのメモリ解放
  cudaFree(dA.elements);
  cudaFree(dB.elements);
  cudaFree(dC.elements);
  return C;
}

matrix matrixMul_cpu(matrix A, matrix B){
  //計算結果行列の用意
  matrix C;
  C.height = A.height;
  C.width = B.width;
  C.elements = new float[C.height*C.width];

  for(int i = 0; i < C.height; i++){
    for(int j = 0; j < C.width; j++){
      float x = 0.0f;
      for (int k = 0; k < A.width; k++) {
        x += A.elements[i*A.width+k]*B.elements[k*B.width+j];
      }

      C.elements[i*C.width+j] = x;
    }
  }

  return C;
}

void randomInit(float* data, int size, float maxVal){
  for(int i = 0; i < size; i++){
    data[i] = maxVal*(rand()/(float)RAND_MAX);
  }
}

int mainmat(){
  matrix A, B, C;
  A.height = A.width = SIZE_RATE*BLOCK_SIZE;
  B.height = B.width = SIZE_RATE*BLOCK_SIZE;

  A.elements = new float[A.width*A.height];
  B.elements = new float[B.width*B.height];

  randomInit(A.elements, A.width*A.height, 10);
  randomInit(B.elements, B.width*B.height, 10);

  //計測開始
  clock_t start = clock();
  C = matrixMul_cpu(A, B);
  //計測終了
  clock_t end = clock();
  double rate = (double)(end - start);
  std::cout << "cpuMultime = " << rate / CLOCKS_PER_SEC << "sec.\n";

  delete [] C.elements;

  //計測開始
  start = clock();
  C = matrixMul_gpu(A, B);
  //計測終了
  end = clock();
  std::cout << "gpuMultime = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
  std::cout << "rate = " << rate/((double)(end - start)) << std::endl;
  //printMatrix(A);
  //printMatrix(B);
  //printMatrix(C);

  // ホストメモリ解放
  delete [] A.elements;
  delete [] B.elements;
  delete [] C.elements;

  return 0;
}
