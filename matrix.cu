#include <iostream>
#include <cstdio>

#define BLOCK_SIZE 16
#define SIZE_RATE 10

struct matrix{
  int height;
  int width;
  double *elements;
};

void printShape(matrix &m, std::string name){
  std::cout << name << ": " << m.height << ", " << m.width << std::endl;
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

__global__ void matrixCpy_cuda(matrix M, matrix org){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] = org.elements[row*org.width+col];
  }
}

static void matrixCpy(matrix& d_m_in, matrix& d_m_ac){
  //入力のサイズをコピー元と合わせる。
  cudaFree(d_m_in.elements);
  d_m_in.height = d_m_ac.height; d_m_in.width = d_m_ac.width;
  cudaMalloc((void**)&d_m_in.elements, d_m_in.height*d_m_in.width*sizeof(double));
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixCpy_cuda<<<gld, blk>>>(d_m_in, d_m_ac);
}

__global__ void matrixAdd_cuda(matrix M, matrix add){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] += add.elements[row*add.width+col];
  }
}

static void matrixAdd(matrix& d_m_in, matrix& d_m_ac){
  if(d_m_in.height != d_m_ac.height || d_m_in.width != d_m_ac.width){
    std::cout << "add err." << '\n';
    return;
  }
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixAdd_cuda<<<gld, blk>>>(d_m_in, d_m_ac);
}

__global__ void matrixMinus_cuda(matrix M, matrix minus){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] -= minus.elements[row*minus.width+col];
  }
}

static void matrixMinus(matrix& d_m_in, matrix& d_m_ac){
  if(d_m_in.height != d_m_ac.height || d_m_in.width != d_m_ac.width){
    std::cout << "minus err." << '\n';
    return;
  }
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixMinus_cuda<<<gld, blk>>>(d_m_in, d_m_ac);
}

__global__ void matrixConstMul_cuda(matrix M, double rate){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] *= rate;
  }
}

static void matrixConstMul(matrix& d_m_in, double rate){
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixConstMul_cuda<<<gld, blk>>>(d_m_in, rate);
}

__global__ void matrixMul_cuda(matrix A, matrix B, matrix C){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < C.height && col < C.width){
    double x = 0.0f;
    for (int i = 0; i < A.width; i++) {
      x += A.elements[row*A.width+i]*B.elements[i*B.width+col];
    }

    C.elements[row*C.width+col] = x;
  }
}

static void matrixMul(matrix& d_m_in, matrix& d_m_ac){
  if(d_m_in.width != d_m_ac.height){
    std::cout << "mul err." << '\n';
    return;
  }
  //デバイスに演算結果の領域を確保
  matrix d_ans;
  d_ans.width = d_m_ac.width; d_ans.height = d_m_in.height;

  int size;
  //デバイスにメモリ確保
  size = d_ans.width*d_ans.height*sizeof(double);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_ans.width-1+blk.x)/blk.x, (d_ans.height-1+blk.y)/blk.y);

  matrixMul_cuda<<<gld, blk>>>(d_m_in, d_m_ac, d_ans);

  //不要になった入力のメモリの開放
  cudaFree(d_m_in.elements);

  //演算結果を引き継ぐ
  d_m_in = d_ans;
}

__global__ void matrixAddBias_cuda(matrix M, matrix bias){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] += bias.elements[col];
  }
}

static void matrixAddBias(matrix& d_m_in, matrix& d_m_ac){
  if(d_m_in.width != d_m_ac.width){
    std::cout << "bias err." << '\n';
    return;
  }
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixAddBias_cuda<<<gld, blk>>>(d_m_in, d_m_ac);
}

__global__ void matrixRelu_cuda(matrix M){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] = (M.elements[row*M.width+col] < 0)? 0: M.elements[row*M.width+col];
  }
}

static void matrixRelu(matrix& d_m_in){
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixRelu_cuda<<<gld, blk>>>(d_m_in);
}

__global__ void matrixSoftmax_cuda(matrix M){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  //計算が必要なスレッドか確認
  if(row < M.height){
    double sum = 0;
    double max = M.elements[row*M.width];
    double min = M.elements[row*M.width];
    for (int j = 0; j < M.width; j++) {
        max = (M.elements[row*M.width+j] > max)? M.elements[row*M.width+j]: max;
        min = (M.elements[row*M.width+j] < min)? M.elements[row*M.width+j]: min;
    }
    double mid = (max + min)/2;
    for (int j = 0; j < M.width; j++) {
        sum += exp(M.elements[row*M.width+j] - mid);
    }
    for (int j = 0; j < M.width; j++) {
        M.elements[row*M.width+j] = exp(M.elements[row*M.width+j] - mid)/sum;
    }
  }
}

static void matrixSoftmax(matrix& d_m_in){
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(1, BLOCK_SIZE);
  dim3 gld(1, (d_m_in.height-1+blk.y)/blk.y);

  matrixSoftmax_cuda<<<gld, blk>>>(d_m_in);
}

__global__ void matrixReluWithOther_cuda(matrix M, matrix relufrom){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < M.height && col < M.width){
    M.elements[row*M.width+col] = (relufrom.elements[row*relufrom.width+col] < 0)? 0: M.elements[row*M.width+col];
  }
}

static void matrixReluWithOther(matrix& d_m_in, matrix& d_m_ac){
  if(d_m_in.height != d_m_ac.height || d_m_in.width != d_m_ac.width){
    std::cout << "relu with other err." << '\n';
    return;
  }
  //入力のサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_m_in.width-1+blk.x)/blk.x, (d_m_in.height-1+blk.y)/blk.y);

  matrixReluWithOther_cuda<<<gld, blk>>>(d_m_in, d_m_ac);
}

__global__ void matrixTranspose_cuda(matrix M, matrix trans){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(row < trans.height && col < trans.width){
    trans.elements[row*trans.width+col] = M.elements[col*M.width+row];
  }
}

static void matrixTranspose(matrix& d_m_in){
  //デバイスに演算結果の領域を確保
  matrix d_ans;
  d_ans.height = d_m_in.width; d_ans.width = d_m_in.height;
  int size = d_ans.width*d_ans.height*sizeof(double);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((d_ans.width-1+blk.x)/blk.x, (d_ans.height-1+blk.y)/blk.y);

  matrixTranspose_cuda<<<gld, blk>>>(d_m_in, d_ans);

  //不要になった入力のメモリの開放
  cudaFree(d_m_in.elements);

  //演算結果を引き継ぐ
  d_m_in = d_ans;
}

__global__ void matrixSumColumn_cuda(matrix M, matrix ans){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  //計算が必要なスレッドか確認
  if(col < ans.width){
    double x = 0;
    for(int i = 0; i < M.height; i++){
      x += M.elements[i*M.width+col];
    }
    ans.elements[col] = x;
  }
}

static void matrixSumColumn(matrix& d_m_in){
  //デバイスに演算結果の領域を確保
  matrix d_ans;
  d_ans.height = 1; d_ans.width = d_m_in.width;
  int size = d_ans.width*d_ans.height*sizeof(double);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, 1);
  dim3 gld((d_ans.width-1+blk.x)/blk.x, 1);

  matrixSumColumn_cuda<<<gld, blk>>>(d_m_in, d_ans);

  //不要になった入力のメモリの開放
  cudaFree(d_m_in.elements);

  //演算結果を引き継ぐ
  d_m_in = d_ans;
}

__global__ void matrixSumRow_cuda(matrix M, matrix ans){
  //行列Mにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  //計算が必要なスレッドか確認
  if(row < ans.height){
    double x = 0;
    for(int i = 0; i < M.width; i++){
      x += M.elements[row*M.width+i];
    }
    ans.elements[row] = x;
  }
}

static void matrixSumRow(matrix& d_m_in){
  //デバイスに演算結果の領域を確保
  matrix d_ans;
  d_ans.height = d_m_in.height; d_ans.width = 1;
  int size = d_ans.width*d_ans.height*sizeof(double);
  cudaMalloc((void**)&d_ans.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(1, BLOCK_SIZE);
  dim3 gld(1, (d_ans.height-1+blk.y)/blk.y);

  matrixSumRow_cuda<<<gld, blk>>>(d_m_in, d_ans);

  //不要になった入力のメモリの開放
  cudaFree(d_m_in.elements);

  //演算結果を引き継ぐ
  d_m_in = d_ans;
}

__global__ void matrixCrossE_cuda(matrix err, matrix result, matrix teacher){
  //行列Cにおけるどこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  int idx = row*err.width+col;
  //計算が必要なスレッドか確認
  if(row < err.height && col < err.width){
    err.elements[idx] = -teacher.elements[idx]*log(result.elements[idx]);
  }
}

static void matrixCrossE(matrix& err, matrix& result, matrix& teacher){
  if(result.height != teacher.height || result.width != teacher.width){
    std::cout << "cross ent err." << '\n';
    return;
  }
  //デバイスに演算結果の領域を確保
  err.width = result.width; err.height = result.height;

  int size = err.width*err.height*sizeof(double);
  cudaMalloc((void**)&err.elements, size);

  //Cのサイズに合わせてブロックとグリッドの設定
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((err.width-1+blk.x)/blk.x, (err.height-1+blk.y)/blk.y);

  matrixCrossE_cuda<<<gld, blk>>>(err, result, teacher);

}

__global__ void matrixAdam_cuda(double leaning_rate, int sequence, matrix ada_grad, matrix velocity_matrix, matrix prime_w_list, matrix w_list){
  //cudaの処理
  //どこを計算するスレッドか確定する。
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  int idx = row*w_list.width + col;

  //計算が必要なスレッドか確認
  if(row < w_list.height && col < w_list.width){
    velocity_matrix.elements[idx] = 0.9*velocity_matrix.elements[idx] + 0.1*prime_w_list.elements[idx];
    ada_grad.elements[idx] = 0.999*ada_grad.elements[idx] + 0.001*prime_w_list.elements[idx]*prime_w_list.elements[idx];
    double v_hat = velocity_matrix.elements[idx]/(1 - pow(0.9, sequence));
    double a_hat = ada_grad.elements[idx]/(1 - pow(0.999, sequence));
    w_list.elements[idx] -= (leaning_rate*v_hat)/(sqrt(a_hat)+0.00000001);
  }
}

static void matrixAdam(double leaning_rate, int sequence, matrix& ada_grad, matrix& velocity_matrix, matrix& prime_w_list, matrix& w_list){
  //デバイスでの処理
  dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gld((w_list.width-1+blk.x)/blk.x, (w_list.height-1+blk.y)/blk.y);

  matrixAdam_cuda<<<gld, blk>>>(leaning_rate, sequence, ada_grad, velocity_matrix, prime_w_list, w_list);
}

void randomInit(matrix m, int maxVal){
  for(int i = 0; i < m.height*m.width; i++) m.elements[i] = int(maxVal*(double(rand())/RAND_MAX)) - maxVal/2.0;
}

void checkFunction(void (*func)(matrix&, matrix&), int ah, int aw, int bh, int bw){
  matrix A, B;
  A.height = ah; A.width = aw;
  B.height = bh; B.width = bw;

  A.elements = new double[A.width*A.height];
  B.elements = new double[B.width*B.height];

  randomInit(A, 10);
  randomInit(B, 10);

  //演算前確認
  std::cout << "matrix:in(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);
  std::cout << "matrix:act(" << B.height << ", " << B.width << ") =" << std::endl;
  printMatrix(B);

  matrix dA, dB;
  dA.width = A.width; dA.height = A.height;
  dB.width = B.width; dB.height = B.height;

  int size = dA.width*dA.height*sizeof(double);
  cudaMalloc((void**)&dA.elements, size);
  cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

  size = dB.width*dB.height*sizeof(double);
  cudaMalloc((void**)&dB.elements, size);
  cudaMemcpy(dB.elements, B.elements, size, cudaMemcpyHostToDevice);

  func(dA, dB);

  //Aのサイズを変更されたdAのサイズに合わせる。
  delete [] A.elements;
  A.height = dA.height;
  A.width = dA.width;
  A.elements = new double[A.height*A.width];
  size = dA.width*dA.height*sizeof(double);
  cudaMemcpy(A.elements, dA.elements, size, cudaMemcpyDeviceToHost);

  std::cout << "matrix:ans(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);

  // ホストメモリ解放
  delete [] A.elements;
  delete [] B.elements;

  for(int i = 0; i < 25; i++) std::cout << "-";
  std::cout << std::endl;
}

void checkFunction2(void (*func)(matrix&), int ah, int aw){
  //行列作成
  matrix A;
  A.height = ah; A.width = aw;
  A.elements = new double[A.width*A.height];
  randomInit(A, 10);

  //演算前確認
  std::cout << "matrix:in(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);

  matrix dA;
  dA.width = A.width; dA.height = A.height;

  int size = dA.width*dA.height*sizeof(double);
  cudaMalloc((void**)&dA.elements, size);
  cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

  func(dA);

  //Aのサイズを変更されたdAのサイズに合わせる。
  delete [] A.elements;
  A.height = dA.height;
  A.width = dA.width;
  A.elements = new double[A.height*A.width];
  size = dA.width*dA.height*sizeof(double);
  cudaMemcpy(A.elements, dA.elements, size, cudaMemcpyDeviceToHost);

  std::cout << "matrix:ans(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);

  // ホストメモリ解放
  delete [] A.elements;

  for(int i = 0; i < 25; i++) std::cout << "-";
  std::cout << std::endl;
}

void checkFunction3(void (*func)(matrix&, double), int ah, int aw, double rate){
  //行列作成
  matrix A;
  A.height = ah; A.width = aw;
  A.elements = new double[A.width*A.height];
  randomInit(A, 10);

  //演算前確認
  std::cout << "matrix:in(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);

  matrix dA;
  dA.width = A.width; dA.height = A.height;

  int size = dA.width*dA.height*sizeof(double);
  cudaMalloc((void**)&dA.elements, size);
  cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

  func(dA, rate);

  //Aのサイズを変更されたdAのサイズに合わせる。
  delete [] A.elements;
  A.height = dA.height;
  A.width = dA.width;
  A.elements = new double[A.height*A.width];
  size = dA.width*dA.height*sizeof(double);
  cudaMemcpy(A.elements, dA.elements, size, cudaMemcpyDeviceToHost);

  std::cout << "matrix:ans(" << A.height << ", " << A.width << ") =" << std::endl;
  printMatrix(A);

  // ホストメモリ解放
  delete [] A.elements;

  for(int i = 0; i < 25; i++) std::cout << "-";
  std::cout << std::endl;
}

void checkAll(){
  srand((unsigned int)time(0));
  std::cout << "cpy" << std::endl;
  checkFunction(matrixCpy, 3,3,2,2);
  std::cout << "add" << std::endl;
  checkFunction(matrixAdd, 2,2,2,2);
  std::cout << "minus" << std::endl;
  checkFunction(matrixMinus, 2,2,2,2);
  std::cout << "mul" << std::endl;
  checkFunction(matrixMul, 2,3,3,2);
  std::cout << "bias" << std::endl;
  checkFunction(matrixAddBias, 2,3,1,3);
  std::cout << "reluwithother" << std::endl;
  checkFunction(matrixReluWithOther, 2,3,2,3);
  std::cout << "relu" << std::endl;
  checkFunction2(matrixRelu, 2,3);
  std::cout << "softmax" << std::endl;
  checkFunction2(matrixSoftmax, 2,3);
  std::cout << "trans" << std::endl;
  checkFunction2(matrixTranspose, 2,3);
  std::cout << "sumcol" << std::endl;
  checkFunction2(matrixSumColumn, 3,4);
  std::cout << "sumrow" << std::endl;
  checkFunction2(matrixSumRow, 4,3);
  std::cout << "const Mul " << 2 << std::endl;
  checkFunction3(matrixConstMul, 2,2,2);//最後の引数は倍率ß
}

//int main(){
//  checkAll();
//}
