#include <iostream>
#include <vector>
#include <fstream>
#include <string>

typedef unsigned int uint;
typedef float Scalar;

void genData(std::string filename)
{
    std::ofstream file1(filename + "-in.csv");
    std::ofstream file2(filename + "-out.csv");
    for (uint r = 0; r < 1000; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        Scalar y = rand() / Scalar(RAND_MAX);
        file1 << x << ", " << y << std::endl;
        file2 << 2 * x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}

int main() {
    genData("test");
}