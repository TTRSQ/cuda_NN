#include "matrix.cu"
#include "stringpp.hpp"
#include <cstdio>
#include <ctime>
#include <random>
#include <fstream>
#include <map>

class matrix_set{
  bool host_f;
  bool device_f;
public:
  matrix host;
  matrix device;

  matrix_set(){
    host_f = false;
    device_f = false;
  }

  bool hostExist(){
    return host_f;
  }

  bool deviceExist(){
    return device_f;
  }

  void resizeHost(int h, int w){
    host.height = h; host.width = w;
    if(host_f){
      delete [] host.elements;
    }
    host.elements = new double[h*w];
    host_f = true;
  }

  void resizeHostWith0(int h, int w){
    host.height = h; host.width = w;
    if(host_f){
      delete [] host.elements;
    }
    host.elements = new double[h*w];
    for(int i = 0; i < h*w; i++) host.elements[i] = 0;
    host_f = true;
  }

  void deleteHost(){
    if(host_f){
      delete [] host.elements;
      host_f = false;
    }
  }

  void resizeDevice(int h, int w){
    device.height = h; device.width = w;
    if(device_f){
      cudaFree(device.elements);
    }
    cudaMalloc((void**)&device.elements, h*w*sizeof(double));
    device_f = true;
  }

  void deleteDevice(){
    if(device_f){
      cudaFree(device.elements);
      device_f = false;
    }
  }

  void deleteBoth(){
    deleteDevice();
    deleteHost();
  }

  void cpy_to_device(){
    if(host_f){
      resizeDevice(host.height, host.width);
      cudaMemcpy(device.elements, host.elements, host.height*host.width*sizeof(double), cudaMemcpyHostToDevice);
    }else{
      std::cout << "不正なコピーが起きました。" << std::endl;
    }
  }

  void cpy_to_host(){
    if(device_f){
      resizeHost(device.height, device.width);
      cudaMemcpy(host.elements, device.elements, device.height*device.width*sizeof(double), cudaMemcpyDeviceToHost);
    }else{
      std::cout << "不正なコピーが起きました。" << std::endl;
    }
  }
};

class affine_relu{
public:
    matrix_set w_list;
    matrix_set prime_w_list;
    matrix_set velocity_matrix;
    matrix_set ada_grad;
    matrix_set relu_prime;

    matrix_set x_transpose;

    matrix_set bias;
    matrix_set bias_prime;
    matrix_set bias_velocity_matrix;
    matrix_set bias_ada_grad;

    affine_relu(){}

    void print_size(){
      //ホスト
        std::cout << "affine(" << w_list.host.height << ", " << w_list.host.width << ")" << std::endl;
        std::cout << "bias(" << bias.host.height << ", " << bias.host.width << ")" << std::endl;
    }

    void push_to_cuda(){
      w_list.cpy_to_device();
      prime_w_list.cpy_to_device();
      velocity_matrix.cpy_to_device();
      ada_grad.cpy_to_device();

      bias.cpy_to_device();
      bias_prime.cpy_to_device();
      bias_velocity_matrix.cpy_to_device();
      bias_ada_grad.cpy_to_device();
    }

    void pull_from_cuda(){
      w_list.cpy_to_host();
      bias.cpy_to_host();
    }

    void purgeMem(){
      w_list.deleteBoth();
      prime_w_list.deleteBoth();
      velocity_matrix.deleteBoth();
      ada_grad.deleteBoth();
      bias.deleteBoth();
      bias_prime.deleteBoth();
      bias_velocity_matrix.deleteBoth();
      bias_ada_grad.deleteBoth();
    }

    void init_with_he(int h, int w, std::default_random_engine engine, std::normal_distribution<> dist){
      //ホストでの処理
      w_list.resizeHost(h, w);
      prime_w_list.resizeHost(h, w);
      velocity_matrix.resizeHost(h, w);
      ada_grad.resizeHost(h, w);

      double s = sqrt(2.0/h);
      for (int i = 0; i < h*w; i++) {
              w_list.host.elements[i] = s*dist(engine);
      }

      bias.resizeHost(1, w);
      bias_prime.resizeHost(1, w);
      bias_velocity_matrix.resizeHost(1, w);
      bias_ada_grad.resizeHost(1, w);

      push_to_cuda();
    }

    void init_size(){
      //ホストでの処理
      if(w_list.hostExist()){
        int h = w_list.host.height; int w = w_list.host.width;
        prime_w_list.resizeHostWith0(h, w);
        velocity_matrix.resizeHostWith0(h, w);
        ada_grad.resizeHostWith0(h, w);

        bias_prime.resizeHostWith0(1, w);
        bias_velocity_matrix.resizeHostWith0(1, w);
        bias_ada_grad.resizeHostWith0(1, w);
      }else{
        std::cout << "イニシャライズできませんでした。" << std::endl;
      }
      push_to_cuda();
    }

    void forward(matrix &in){
      //デバイスでの処理
        matrixCpy(x_transpose.device, in);
        matrixTranspose(x_transpose.device);
        matrixMul(in, w_list.device);
        matrixAddBias(in, bias.device);
        matrixCpy(relu_prime.device, in);
        matrixRelu(in);
    }

    void backward(matrix &in){
      //デバイスでの処理
        //relu
        matrixReluWithOther(in, relu_prime.device);

        //bias
        matrixCpy(bias_prime.device, in);
        matrixSumColumn(bias_prime.device);

        //affine
        matrixMul(x_transpose.device, in);
        matrixCpy(prime_w_list.device, x_transpose.device);

        matrix wtrans;
        matrixCpy(wtrans, w_list.device);
        matrixTranspose(wtrans);
        matrixMul(in, wtrans);
        cudaFree(wtrans.elements);
    }
};

class softmax_cee : public affine_relu{
public:
    matrix_set softmax_output;

    softmax_cee(){}

    void purgeMemS(){
      purgeMem();
      softmax_output.deleteBoth();
    }

    void softmax(matrix &in){
      //デバイス
        matrixSoftmax(in);
    }

    void cross_entropy(matrix& err, matrix &result, matrix &teacher){
      //デバイス、新しく作ったデバイス側のポインタを返す。
        matrixCrossE(err, result, teacher);
    }

    void softmax_forward(matrix &in){
      //デバイス
        matrixCpy(x_transpose.device, in);
        matrixTranspose(x_transpose.device);
        matrixMul(in, w_list.device);
        matrixAddBias(in, bias.device);
        //- relu
        softmax(in);
        matrixCpy(softmax_output.device, in);
    }

    void softmax_backward(matrix &in){
      //デバイス
        matrixMinus(in, softmax_output.device);
        matrixConstMul(in, -1);

        //bias
        matrixCpy(bias_prime.device, in);
        matrixSumColumn(bias_prime.device);

        //affine
        matrixMul(x_transpose.device, in);
        matrixCpy(prime_w_list.device, x_transpose.device);

        matrix wtrans;
        matrixCpy(wtrans, w_list.device);
        matrixTranspose(wtrans);
        matrixMul(in, wtrans);
        cudaFree(wtrans.elements);
    }
};

class network{
public:
    std::vector<affine_relu> affine;
    softmax_cee softmax;
    int input, hide, hide_neuron, output, mini_badge;

    network(){}

    network(int _input, int _hide, int _hide_neuron, int _output, int _mini_badge){
      //ホスト
        input = _input;
        hide = _hide;
        hide_neuron = _hide_neuron;
        output = _output;
        mini_badge = _mini_badge;

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0);

        affine.resize(hide);
        affine[0].init_with_he(input, hide_neuron, engine, dist);
        for (int i = 1; i < hide; i++) {
            affine[i].init_with_he(hide_neuron, hide_neuron, engine, dist);
        }
        softmax.init_with_he(hide_neuron, output, engine, dist);
    }

    void purgeNet(){
      for(int i = 0; i < affine.size(); i++) affine[i].purgeMem();
      softmax.purgeMemS();
    }

    void save_network(std::string name){
      //ホスト
        for(int i = 0; i < affine.size(); i++){
          affine[i].pull_from_cuda();
        }
        softmax.pull_from_cuda();

        name = "NNpram/" + name + ".txt";
        std::ofstream outputfile(name);

        outputfile << 7 << std::endl;
        time_t ti = time(NULL);
        outputfile << ctime(&ti);
        outputfile << "input " << input << std::endl;
        outputfile << "hide " << hide << std::endl;
        outputfile << "hide_neuron " << hide_neuron << std::endl;
        outputfile << "output " << output << std::endl;
        outputfile << "mini_badge " << mini_badge << std::endl;

        outputfile << "affine " << affine.size() << std::endl;
        for (int i = 0; i < affine.size(); i++) {
            outputfile << affine[i].w_list.host.height << ' ' << affine[i].w_list.host.width << std::endl;
            for (int j = 0; j < affine[i].w_list.host.height; j++) {
                for (int k = 0; k < affine[i].w_list.host.width; k++) {
                    if (k != 0) outputfile << ' ';
                    outputfile << affine[i].w_list.host.elements[j*affine[i].w_list.host.width+k];
                }
                outputfile << std::endl;
                if (j == affine[i].w_list.host.height - 1) {
                    outputfile << "bias " << affine[i].bias.host.width << std::endl;
                    for (int bia = 0; bia < affine[i].bias.host.width; bia++) {
                        if(bia != 0) outputfile << ' ';
                        outputfile << affine[i].bias.host.elements[bia];
                    }
                    outputfile << std::endl;
                }
            }
        }
        outputfile << "softmax" << std::endl;
        outputfile << softmax.w_list.host.height << ' ' << softmax.w_list.host.width << std::endl;
        for (int i = 0; i < softmax.w_list.host.height; i++) {
            for (int j = 0; j < softmax.w_list.host.width; j++) {
                if(j != 0) outputfile << ' ';
                outputfile << softmax.w_list.host.elements[i*softmax.w_list.host.width+j];
            }
            outputfile << std::endl;
        }

        outputfile << "bias " << softmax.bias.host.width << std::endl;
        for (int bia = 0; bia < softmax.bias.host.width; bia++) {
            if(bia != 0) outputfile << ' ';
            outputfile << softmax.bias.host.elements[bia];
        }

        purgeNet();

        outputfile.close();
    }

    void load_network(std::string name){
      //ホストでの処理
        name = "NNpram/" + name + ".txt";
        std::ifstream inputfile(name);
        std::vector<std::string> buff;
        std::string str;
        getline(inputfile, str);
        int N = std::stoi(str);
        std::map<std::string, int> map;

        for (int i = 0; i < N; i++) {
            getline(inputfile, str);
            if(i != 0){//1行目は時間なので捨てる。
                buff = split(str, ' ');
                map.insert(std::make_pair(buff[0], std::stoi(buff[1])));
            }
        }

        input = map.at("input");
        hide = map.at("hide");
        hide_neuron = map.at("hide_neuron");
        output = map.at("output");
        mini_badge = map.at("mini_badge");
        affine.resize(map.at("affine"));

        //affine
        for (int i = 0; i < map.at("affine"); i++) {
            int h, w;
            getline(inputfile, str);
            buff = split(str, ' ');
            h = std::stoi(buff[0]);
            w = std::stoi(buff[1]);
            affine[i].w_list.resizeHost(h, w);
            for (int j = 0; j < h; j++) {
                getline(inputfile, str);
                buff = split(str, ' ');
                for (int k = 0; k < w; k++) {
                    affine[i].w_list.host.elements[j*affine[i].w_list.host.width+k] = std::stod(buff[k]);
                }
            }
            //bias
            getline(inputfile, str);
            buff = split(str, ' ');
            int biasnum = std::stoi(buff[1]);
            affine[i].bias.resizeHost(1, biasnum);
            affine[i].init_size();
            getline(inputfile, str);
            buff = split(str, ' ');
            for (int j = 0; j < biasnum; j++) {
                affine[i].bias.host.elements[j] = std::stod(buff[j]);
            }
            affine[i].push_to_cuda();
        }
        getline(inputfile, str);

        //softmax
        int h, w;
        getline(inputfile, str);
        buff = split(str, ' ');
        h = std::stoi(buff[0]);
        w = std::stoi(buff[1]);
        softmax.w_list.resizeHost(h, w);
        for (int j = 0; j < h; j++) {
            getline(inputfile, str);
            buff = split(str, ' ');
            for (int k = 0; k < w; k++) {
                softmax.w_list.host.elements[j*softmax.w_list.host.width+k] = std::stod(buff[k]);
            }
        }

        //bias
        getline(inputfile, str);
        buff = split(str, ' ');
        int biasnum = std::stoi(buff[1]);
        softmax.bias.resizeHost(1, biasnum);
        getline(inputfile, str);
        buff = split(str, ' ');
        for (int j = 0; j < biasnum; j++) {
            softmax.bias.host.elements[j] = std::stod(buff[j]);
        }
        softmax.init_size();
        softmax.push_to_cuda();
    }

    std::vector<double> prediction(std::vector<double> &in){
      //inをホストから受け取ってデバイスにコピー
      matrix_set m_in;
      int h = 1; int w = in.size();
      m_in.resizeHost(h, w);
      for(int i = 0; i < w; i++) m_in.host.elements[i] = in[i];
      m_in.cpy_to_device();

      //順伝搬
      for(int i = 0; i < affine.size(); i++){
        affine[i].forward(m_in.device);
      }
      matrixMul(m_in.device, softmax.w_list.device);
      std::vector<double> v;
      m_in.cpy_to_host();
      for(int i = 0; i < w; i++) v.push_back(m_in.host.elements[i]);

      m_in.deleteBoth();
      return v;
    }

    double calculate_error(std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &teacher){
      double err;

      matrix_set m_in, m_teacher;
      int h = in.size(); int w = in[0].size();
      m_in.resizeHost(h, w);
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
          m_in.host.elements[i*w+j] = in[i][j];
        }
      }
      m_in.cpy_to_device();

      h = teacher.size(); w = teacher[0].size();
      m_teacher.resizeHost(h, w);
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
          m_teacher.host.elements[i*w+j] = teacher[i][j];
        }
      }
      m_teacher.cpy_to_device();

      for(int i = 0; i < affine.size(); i++){
          affine[i].forward(m_in.device);
      }
      softmax.softmax_forward(m_in.device);

      matrix m_err;
      softmax.cross_entropy(m_err, m_in.device, m_teacher.device);

      matrixSumColumn(m_err);
      matrixSumRow(m_err);

      cudaMemcpy(&err, m_err.elements, sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(m_err.elements);
      m_in.deleteBoth();
      m_teacher.deleteBoth();

      return err/h;
    }

    void for_and_backward(std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &teacher){
      //ホストから受け取ってデバイス処理。
      matrix_set d_in, d_teacher;
      int h = in.size(); int w = in[0].size();
      d_in.resizeHost(h, w);
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
          d_in.host.elements[w*i+j] = in[i][j];
        }
      }
      d_in.cpy_to_device();

      h = teacher.size(); w = teacher[0].size();
      d_teacher.resizeHost(h, w);
      for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
          d_teacher.host.elements[w*i+j] = teacher[i][j];
        }
      }
      d_teacher.cpy_to_device();

      for(int i = 0; i < affine.size(); i++){
          affine[i].forward(d_in.device);
      }

      softmax.softmax_forward(d_in.device);
      softmax.softmax_backward(d_teacher.device);

      for(int i = affine.size() - 1; i >= 0; i--){
          affine[i].backward(d_teacher.device);
      }

      d_in.deleteBoth();
      d_teacher.deleteBoth();
    }

    static void adam(double leaning_rate, int sequence, matrix& ada_grad, matrix& velocity_matrix, matrix& prime_w_list, matrix& w_list){
      matrixAdam(leaning_rate, sequence, ada_grad, velocity_matrix, prime_w_list, w_list);
    }

    void leaning(double leaning_rate, int sequence, void (*func)(double, int, matrix&, matrix&, matrix&, matrix&)){
      //デバイスでの処理
        for (int i = 0; i < affine.size(); i++) {
            func(leaning_rate, sequence, affine[i].ada_grad.device, affine[i].velocity_matrix.device, affine[i].prime_w_list.device, affine[i].w_list.device);
            func(leaning_rate, sequence, affine[i].bias_ada_grad.device, affine[i].bias_velocity_matrix.device, affine[i].bias_prime.device, affine[i].bias.device);
        }
        func(leaning_rate, sequence, softmax.ada_grad.device, softmax.velocity_matrix.device, softmax.prime_w_list.device, softmax.w_list.device);
        func(leaning_rate, sequence, softmax.bias_ada_grad.device, softmax.bias_velocity_matrix.device, softmax.bias_prime.device, softmax.bias.device);
    }

    void leaning_adam(double leaning_rate, int sequence){
        leaning(leaning_rate, sequence, adam);
    }
};
