#ifndef STRINGPP_HPP
#define STRINGPP_HPP

#include <vector>

std::vector<std::string> split(std::string str, char dem){
    std::vector<std::string> vec;
    std::string temp;
    for (int i = 0; i < str.size(); i++) {
        if (str[i] != dem) {
            temp = temp + str[i];
        }else{
            vec.push_back(temp);
            temp = "";
        }
    }
    vec.push_back(temp);
    return vec;
}


#endif // STRINGPP_HPP
