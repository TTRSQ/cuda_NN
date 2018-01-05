#include "matrix.cu"
#include "stringpp.hpp"
#include <cstdio>
#include <ctime>
#include <random>
#include <fstream>
#include <map>

class matrix_set{
public:
  matrix host;
  matrix device;

  void cpy_to_device(){
    cudaFree(device.elements);
    int size = (host.height)*(host.width)*sizeof(float);
    cudaMalloc((void**)&device.elements, size);
    cudaMemcpy(device.elements, host.elements, size, cudaMemcpyHostToDevice);
  }

  void cpy_to_host(){
    delete [] host.elements;
    int size = device.height * device.width;
    host.elements = new float[size];
    size *= sizeof(float);
    cudaMemcpy(host.elements, device.elements, size, cudaMemcpyDeviceToHost);
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

    void init_with_he(int h, int w, std::default_random_engine engine, std::normal_distribution<> dist){
      //ホストでの処理
        matrixSizeInit(w_list.host, h, w);
        matrixSizeInit(prime_w_list.host, h, w);
        matrixSizeInit(velocity_matrix.host, h, w);
        matrixSizeInit(ada_grad.host, h, w);

        double s = sqrt(2.0/h);

        for (int i = 0; i < h*w; i++) {
                w_list.host.elements[i] = s*dist(engine);
        }

        matrixSizeInit(bias.host, 1, w);
        matrixSizeInit(bias_prime.host, 1, w);
        matrixSizeInit(bias_velocity_matrix.host, 1, w);
        matrixSizeInit(bias_ada_grad.host, 1, w);
    }

    void init_size(){
      //ホストでの処理
        sizeInitFromMatrix(prime_w_list.host, w_list.host);
        sizeInitFromMatrix(velocity_matrix.host, w_list.host);
        sizeInitFromMatrix(ada_grad.host, w_list.host);

        sizeInitFromMatrix(bias_prime.host, bias.host);
        sizeInitFromMatrix(bias_velocity_matrix.host, bias.host);
        sizeInitFromMatrix(bias_ada_grad.host, bias.host);
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
    matrix_set cross_entropy_list;
    matrix_set softmax_output;

    softmax_cee(){}

    void softmax(matrix &in){
      //デバイス
        for (int i = 0; i < in.h; i++) {
            double sum = 0;
            double max = in.t[i][0];
            double min = in.t[i][0];
            for (int j = 0; j < in.w; j++) {
                max = std::max(in.t[i][j], max);
                min = std::min(in.t[i][j], min);
            }
            double mid = (max + min)/2;
            for (int j = 0; j < in.w; j++) {
                sum += exp(in.t[i][j] - mid);
            }
            for (int j = 0; j < in.w; j++) {
                in.t[i][j] = exp(in.t[i][j] - mid)/sum;
            }
        }
    }

    double cross_entropy(matrix &in, matrix &teacher){
      //ホストで受け取ってデバイス側に
        std::vector<double> v;
        v.resize(in.h);
        std::fill(v.begin(), v.end(), 0.0);
        for (int i = 0; i < in.h; i++) {
            for (int j = 0; j < in.w; j++) {
                v[i] += -teacher.t[i][j]*log(in.t[i][j]);
            }
        }
        return v;
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

    void save_network(std::string name){
      //ホスト
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
            matrixSizeInit(affine[i].w_list.host, h, w);
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
            matrixSizeInit(affine[i].bias.host , 1, biasnum);
            affine[i].init_size();
            getline(inputfile, str);
            buff = split(str, ' ');
            for (int j = 0; j < biasnum; j++) {
                affine[i].bias.host.elements[j] = std::stod(buff[j]);
            }
        }
        getline(inputfile, str);

        //softmax
        int h, w;
        getline(inputfile, str);
        buff = split(str, ' ');
        h = std::stoi(buff[0]);
        w = std::stoi(buff[1]);
        matrixSizeInit(softmax.w_list.host, h, w);
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
        matrixSizeInit(softmax.bias.host, 1, biasnum);
        softmax.init_size();
        getline(inputfile, str);
        buff = split(str, ' ');
        for (int j = 0; j < biasnum; j++) {
            softmax.bias.host.elements[j] = std::stod(buff[j]);
        }
    }

    std::vector<double> prediction(std::vector<double> in){
      //inをホストから受け取ってデバイスにコピー
        std::vector<std::vector<double> > dvec;
        dvec.push_back(in);
        matrix matin(dvec);
        for(int i = 0; i < affine.size(); i++){
            affine[i].forward(matin);
        }
        matrixMul(matin, softmax.w_list);
        return matin.t[0];
    }

    double calculate_error(matrix in, matrix teacher){
      double *d_err = 0;

        for(int i = 0; i < affine.size(); i++){
            affine[i].forward(in);
        }

        softmax.softmax_forward(in);
        std::vector<double> v;
        v = softmax.cross_entropy(in, teacher);

        double mean = 0;
        for (int i = 0; i < v.size(); i++) {
            mean += v[i]/v.size();
        }
        return mean;
    }

    void for_and_backward(matrix in, matrix teacher){
      //デバイスでの処理
        for(int i = 0; i < affine.size(); i++){
            affine[i].forward(in);
        }

        softmax.softmax_forward(in);
        softmax.softmax_backward(teacher);

        for(int i = affine.size() - 1; i >= 0; i--){
            affine[i].backward(teacher);
        }
    }

    static void adam(double leaning_rate, matrix& ada_grad, matrix& velocity_matrix, matrix& prime_w_list, matrix& w_list){
      matrixAdam(leaning_rate, ada_grad, velocity_matrix, prime_w_list, w_list);
    }

    void leaning(double leaning_rate, void (*func)(double, matrix&, matrix&, matrix&, matrix&)){
      //デバイスでの処理
        for (int i = 0; i < affine.size(); i++) {
            func(leaning_rate, affine[i].ada_grad.device, affine[i].velocity_matrix.device, affine[i].prime_w_list.device, affine[i].w_list.device);
            func(leaning_rate, affine[i].bias_ada_grad.device, affine[i].bias_velocity_matrix.device, affine[i].bias_prime.device, affine[i].bias.device);
        }
        func(leaning_rate, softmax.ada_grad.device, softmax.velocity_matrix.device, softmax.prime_w_list.device, softmax.w_list.device);
        func(leaning_rate, softmax.bias_ada_grad.device, softmax.bias_velocity_matrix.device, softmax.bias_prime.device, softmax.bias.device);
    }

    void leaning_adam(double leaning_rate){
        leaning(leaning_rate, adam);
    }
};
