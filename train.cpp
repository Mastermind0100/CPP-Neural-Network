#include "network.cpp"
#include <typeinfo>

typedef std::vector<RowVector*> data;

int main() {
    NeuralNetwork network({2,3,1});
    data input_data, output_data;
    ReadCSV("test-in.csv", input_data);
    ReadCSV("test-out.csv", output_data);    

    network.train(input_data, output_data);
    return 0;
}