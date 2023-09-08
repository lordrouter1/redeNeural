#include "NeuralNetwork.cpp"

int main()
{
    NeuralNetwork nn(2, 1, 4, 1, 0.01);

    std::vector<std::vector<double>> inputValues = {{1,0},{0,1},{1,1},{0,0}};
    std::vector<std::vector<double>> targetValues = {{1},{1},{0},{0}}; // Valores alvo para treinamento
    
    nn.train(inputValues,targetValues,6000000);

    return 0;
}