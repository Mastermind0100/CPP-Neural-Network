#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf Vector;
typedef unsigned int uint;

Scalar activation_function(Scalar x) {
    return tanhf(x);
}

Scalar activation_function_derivative(Scalar x) {
    return 1 - tanhf(x) * tanhf(x);
}

void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
    data.clear();
    std::ifstream file(filename);
    std::string line, word;
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<Scalar> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(std::stof(&word[0])));
    }
    uint cols = parsed_vec.size();
    data.push_back(new RowVector(cols));
    for (uint i = 0; i < cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }

    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                i++;
            }
        }
    }
}

class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<uint> topology, Scalar learning_rate = Scalar(0.005));
        void propogateForward(RowVector& input);
        void propogateBackward(RowVector& output);
        void calculateErrors(RowVector& output);
        void updateWeights();
        void train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data);

        std::vector<RowVector*> neuronLayers;
        std::vector<RowVector*> cacheLayers;
        std::vector<RowVector*> deltas;
        std::vector<Matrix*> weights;
        Scalar learning_rate;

    private:
        std::vector<uint> topology;
};

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate){
    this->topology = topology;
    this->learning_rate = learningRate;
    
    for (uint i = 0; i < topology.size(); i++) {
        if (i == topology.size() - 1) {
            neuronLayers.push_back(new RowVector(topology[i]));
        }
        else {
            neuronLayers.push_back(new RowVector(topology[i] + 1));
        }

        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        if (i != topology.size()-1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        if (i > 0) {
            if (i != topology.size()-1) {
                weights.push_back(new Matrix(topology[i-1]+1, topology[i]+1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i-1], topology[i]) = 1.0;
            }
            else {
                weights.push_back(new Matrix(topology[i-1]+1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
};

void NeuralNetwork::propogateForward(RowVector& input){
    neuronLayers.front()->block(0,0,1,neuronLayers.front()->size()-1) = input;

    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i-1]) * (*weights[i-1]);
        neuronLayers[i]->block(0,0,1,topology[i]).unaryExpr([](Scalar x){return activation_function(x);});
    }
}

void NeuralNetwork::calculateErrors(RowVector& output) {
    (*deltas.back()) = output - (*neuronLayers.back());

    for (uint i = topology.size()-2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights() {
    for (uint i = 0; i < topology.size() - 1; i++) {
        for (uint c = 0; c < weights[i]->cols(); c++) {
            if (i == topology[i]-2 && c == weights[i]->cols()-1) continue;
            for (uint r = 0; r < weights[i]->rows(); r++) {
                weights[i]->coeffRef(r,c) += learning_rate * deltas[i+1]->coeffRef(c) * activation_function_derivative(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
            }
        }
    }
}

void NeuralNetwork::propogateBackward(RowVector& output) {
    calculateErrors(output);
    std::cout<<"Error Calculations Completed!"<<std::endl;
    updateWeights();
    std::cout<<"Weight Updations done!"<<std::endl;
}

void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data) {
    for (uint i = 0; i < input_data.size(); i++) {
        std::cout<<"Epoch "<<i+1<<" has started..."<<std::endl;
        std::cout<<"Input sent to network: "<<*input_data[i]<<std::endl;
        propogateForward(*input_data[i]);
        std::cout<<"Expected Output: "<<*output_data[i]<<std::endl;
        std::cout<<"Output Produced: "<<*neuronLayers.back()<<std::endl;
        propogateBackward(*output_data[i]);
        std::cout<<"Epoch #"<<i+1<<std::endl;
        std::cout<<"Mean Square Error: "<<std::sqrt((*deltas.back()).dot((*deltas.back()))/deltas.back()->size())<<std::endl;
    }
}