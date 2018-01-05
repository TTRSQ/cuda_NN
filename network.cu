S#include "matrix.cu"
#include "stringpp.hpp"
#include <cstdio>
#include <ctime>
#include <random>
#include <fstream>
#include <map>

class affine_relu{
public:
    matrix w_list;
    matrix prime_w_list;
    matrix velocity_matrix;
    matrix ada_grad;
    matrix relu_prime;

    matrix x_transpose;

    matrix bias;
    matrix bias_prime;
    matrix bias_velocity_matrix;
    matrix bias_ada_grad;

    affine_relu(){}

    void print_size(){
        std::cout << "affine(" << w_list.height << ", " << w_list.width << ")" << std::endl;
        std::cout << "bias(" << bias.height << ", " << bias.width << ")" << std::endl;
    }

    void init_with_he(int h, int w, std::default_random_engine engine, std::normal_distribution<> dist){
      //ホストでの処理
        matrixSizeInit(w_list, h, w);
        matrixSizeInit(prime_w_list, h, w);
        matrixSizeInit(velocity_matrix, h, w);
        matrixSizeInit(ada_grad, h, w);

        double s = sqrt(2.0/h);

        for (int i = 0; i < h*w; i++) {
                w_list.elements[i] = s*dist(engine);
        }

        matrixSizeInit(bias, 1, w);
        matrixSizeInit(bias_prime, 1, w);
        matrixSizeInit(bias_velocity_matrix, 1, w);
        matrixSizeInit(bias_ada_grad, 1, w);
    }

    void init_size(){
      //ホストでの処理
        sizeInitFromMatrix(prime_w_list, w_list);
        sizeInitFromMatrix(velocity_matrix, w_list);
        sizeInitFromMatrix(ada_grad, w_list);

        sizeInitFromMatrix(bias_prime, bias);
        sizeInitFromMatrix(bias_velocity_matrix, bias);
        sizeInitFromMatrix(bias_ada_grad, bias);
    }

    void forward(matrix &in){
      //デバイスでの処理
        matrixCpy(x_transpose, in);
        matrixTranspose(x_transpose);
        matrixMul(in, w_list);
        matrixAddBias(in, bias);
        matrixCpy(relu_prime, in);
        matrixRelu(in);
    }

    void backward(matrix &in){
      //デバイスでの処理
        //relu

        // for (int i = 0; i < in.h; i++) {
        //     for (int j = 0; j < in.w; j++) {
        //         in.t[i][j] = (relu_prime.t[i][j] > 0)? in.t[i][j]: 0;
        //     }
        // }
        matrixFunc2(in, relu_prime, [](float& f_in, float& f_relu_prime){
          f_in = (f_relu_prime > 0)? f_in: 0;
        });

        //bias
        bias_prime = in.sum_column();
        for (int i = 0; i < bias_prime.w; i++) {
            bias_velocity_matrix.t[0][i] = 0.9*bias_velocity_matrix.t[0][i] + 0.1*bias_prime.t[0][i];
            bias_ada_grad.t[0][i] = 0.999*bias_ada_grad.t[0][i] + 0.001*bias_prime.t[0][i]*bias_prime.t[0][i];
        }

        //affine
        matrixMul(x_transpose, in);
        matrixCpy(prime_w_list, x_transpose);

        matrix wtrans;
        matrixCpy(wtrans, w_list);
        matrixTranspose(wtrans);
        matrixMul(in, wtrans);
        cudaFree(wtrans.elements);

        for (int i = 0; i < prime_w_list.h; i++) {
            for (int j = 0; j < prime_w_list.w; j++) {
                velocity_matrix.t[i][j] = 0.9*velocity_matrix.t[i][j] + 0.1*prime_w_list.t[i][j];
                ada_grad.t[i][j] = 0.999*ada_grad.t[i][j] + 0.001*prime_w_list.t[i][j]*prime_w_list.t[i][j];
            }
        }
    }
};

class softmax_cee : public affine_relu{
public:
    matrix cross_entropy_list;
    matrix softmax_output;

    softmax_cee(){}

    void softmax(matrix &in){
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

    std::vector<double> cross_entropy(matrix &in, matrix &teacher){
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
        matrixCpy(x_transpose, in);
        matrixTranspose(x_transpose);
        matrixMul(in, w_list);
        matrixAddBias(in, bias);
        //- relu
        softmax(in);
        matrixCpy(softmax_output, in);
    }

    void softmax_backward(matrix &in){
        matrixFunc2(in, softmax_output, [](float& f_in, float& f_so){
          f_in = f_so - f_in;
        });
        //bias
        bias_prime = in.sum_column();
        for (int i = 0; i < bias_prime.w; i++) {
            bias_velocity_matrix.t[0][i] = 0.9*bias_velocity_matrix.t[0][i] + 0.1*bias_prime.t[0][i];
            bias_ada_grad.t[0][i] = 0.999*bias_ada_grad.t[0][i] + 0.001*bias_prime.t[0][i]*bias_prime.t[0][i];
        }
        //affine
        matrixMul(x_transpose, in);
        matrixCpy(prime_w_list, x_transpose);

        matrix wtrans;
        matrixCpy(wtrans, w_list);
        matrixTranspose(wtrans);
        matrixMul(in, wtrans);
        cudaFree(wtrans.elements);

        for (int i = 0; i < prime_w_list.h; i++) {
            for (int j = 0; j < prime_w_list.w; j++) {
                velocity_matrix.t[i][j] = 0.9*velocity_matrix.t[i][j] + 0.1*prime_w_list.t[i][j];
                ada_grad.t[i][j] = 0.999*ada_grad.t[i][j] + 0.001*prime_w_list.t[i][j]*prime_w_list.t[i][j];
            }
        }
    }
};

class network{
public:
    std::vector<affine_relu> affine;
    softmax_cee softmax;
    int input, hide, hide_neuron, output, mini_badge;

    network(){}

    network(int _input, int _hide, int _hide_neuron, int _output, int _mini_badge){
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
            outputfile << affine[i].w_list.height << ' ' << affine[i].w_list.width << std::endl;
            for (int j = 0; j < affine[i].w_list.height; j++) {
                for (int k = 0; k < affine[i].w_list.width; k++) {
                    if (k != 0) outputfile << ' ';
                    outputfile << affine[i].w_list.elements[j*affine[i].w_list.width+k];
                }
                outputfile << std::endl;
                if (j == affine[i].w_list.height - 1) {
                    outputfile << "bias " << affine[i].bias.width << std::endl;
                    for (int bia = 0; bia < affine[i].bias.width; bia++) {
                        if(bia != 0) outputfile << ' ';
                        outputfile << affine[i].bias.elements[bia];
                    }
                    outputfile << std::endl;
                }
            }
        }
        outputfile << "softmax" << std::endl;
        outputfile << softmax.w_list.height << ' ' << softmax.w_list.width << std::endl;
        for (int i = 0; i < softmax.w_list.height; i++) {
            for (int j = 0; j < softmax.w_list.width; j++) {
                if(j != 0) outputfile << ' ';
                outputfile << softmax.w_list.elements[i*softmax.w_list.width+j];
            }
            outputfile << std::endl;
        }

        outputfile << "bias " << softmax.bias.width << std::endl;
        for (int bia = 0; bia < softmax.bias.width; bia++) {
            if(bia != 0) outputfile << ' ';
            outputfile << softmax.bias.elements[bia];
        }

        outputfile.close();
    }

    void load_network(std::string name){
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
            affine[i].w_list.matrixSizeInit(h, w);
            for (int j = 0; j < h; j++) {
                getline(inputfile, str);
                buff = split(str, ' ');
                for (int k = 0; k < w; k++) {
                    affine[i].w_list.t[j][k] = std::stod(buff[k]);
                }
            }
            //bias
            getline(inputfile, str);
            buff = split(str, ' ');
            int biasnum = std::stoi(buff[1]);
            affine[i].bias.matrixSizeInit(1, biasnum);
            affine[i].init_size();
            getline(inputfile, str);
            buff = split(str, ' ');
            for (int j = 0; j < biasnum; j++) {
                affine[i].bias.t[0][j] = std::stod(buff[j]);
            }
        }
        getline(inputfile, str);

        //softmax
        int h, w;
        getline(inputfile, str);
        buff = split(str, ' ');
        h = std::stoi(buff[0]);
        w = std::stoi(buff[1]);
        softmax.w_list.matrixSizeInit(h, w);
        for (int j = 0; j < h; j++) {
            getline(inputfile, str);
            buff = split(str, ' ');
            for (int k = 0; k < w; k++) {
                softmax.w_list.t[j][k] = std::stod(buff[k]);
            }
        }
        //bias
        getline(inputfile, str);
        buff = split(str, ' ');
        int biasnum = std::stoi(buff[1]);
        softmax.bias.matrixSizeInit(1, biasnum);
        softmax.init_size();
        getline(inputfile, str);
        buff = split(str, ' ');
        for (int j = 0; j < biasnum; j++) {
            softmax.bias.t[0][j] = std::stod(buff[j]);
        }
    }

    std::vector<double> prediction(std::vector<double> in){
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

    static void adam(double leaning_rate, matrix &ada_grad, matrix &velocity_matrix, matrix &prime_w_list, matrix &w_list){
      //デバイスでの処理
        // for (int i = 0; i < prime_w_list.h; i++) {
        //     for (int j = 0; j < prime_w_list.w; j++) {
        //         w_list.t[i][j] -= (leaning_rate*10*velocity_matrix.t[i][j])/(sqrt(1000*ada_grad.t[i][j])+0.00000001);
        //     }
        // }
        matrixFunc3(w_list, velocity_matrix, ada_grad, [&leaning_rate](float& f_w, float& f_vm, float& f_ag){
          f_w -= (leaning_rate*10*f_vm)/(sqrt(1000*f_ag)+0.00000001);
        });
    }

    static void sgd(double leaning_rate, matrix &ada_grad, matrix &velocity_matrix, matrix &prime_w_list, matrix &w_list){
      //デバイスでの処理
        // for (int i = 0; i < prime_w_list.h; i++) {
        //     for (int j = 0; j < prime_w_list.w; j++) {
        //         w_list.t[i][j] -= leaning_rate*prime_w_list.t[i][j];
        //     }
        // }
        matrixFunc2(w_list, prime_w_list, [&leaning_rate](float& f_w, float f_wp){
          f_w -= leaning_rate*f_wp;
        });
    }

    void leaning(double leaning_rate, void (*func)(double, matrix&, matrix&, matrix&, matrix&)){
      //デバイスでの処理
        for (int i = 0; i < affine.size(); i++) {
            func(leaning_rate, affine[i].ada_grad, affine[i].velocity_matrix, affine[i].prime_w_list, affine[i].w_list);
            func(leaning_rate, affine[i].bias_ada_grad, affine[i].bias_velocity_matrix, affine[i].bias_prime, affine[i].bias);
        }
        func(leaning_rate, softmax.ada_grad, softmax.velocity_matrix, softmax.prime_w_list, softmax.w_list);
        func(leaning_rate, softmax.bias_ada_grad, softmax.bias_velocity_matrix, softmax.bias_prime, softmax.bias);
    }

    void leaning_adam(double leaning_rate){
        leaning(leaning_rate, adam);
    }

    void leaning_sgd(double leaning_rate){
        leaning(leaning_rate, sgd);
    }
};
