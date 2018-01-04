
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

void matrixMul_gpu(matrix m_in, matrix m_ac){
  matrix ans;
  ans.height = m_in.height;
  ans.width = m_ac.width;
  ans.elements = new float[ans.height*ans.width];
  //デバイス要変数の用意
  matrix d_ans;
  d_ans.width = ans.width; d_ans.height = ans.height;

  int size;
  //デバイスにメモリ確保
  size = d_ans.width*d_ans.height*sizeof(float);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((ans.width-1+blk.x)/blk.x, (ans.height-1+blk.y)/blk.y);

  matrixMul<<<gld, blk>>>(m_in, m_ac, d_ans);

  //不要になった入力のメモリの開放
  cudaFree(m_in.elements);
  
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
