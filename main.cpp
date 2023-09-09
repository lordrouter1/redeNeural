#include "NeuralNetwork.cpp"

int main()
{
    NeuralNetwork nn(8, 1, 6, 4, 0.01);

    std::vector<std::vector<double>> inputValues = {{0,0,0,0,0,0,0,0},{0,0,0,1,0,0,0,0},{0,0,0,0,0,0,0,1},{0,0,0,1,0,0,0,1},{0,0,0,1,0,0,1,0},{0,0,1,0,0,0,0,1}};
    std::vector<std::vector<double>> targetValues = {{0,0,0,0},{0,0,0,1},{0,0,0,1},{0,0,1,0},{0,0,1,1},{0,0,1,1}}; // Valores alvo para treinamento
    
    if(true){
        nn.train(inputValues,targetValues,6000000,0.00001);
    }
    else{
        nn.importDeltas("deltas.out");
        nn.importWeights("weights.out");
    }

    nn.setInput(inputValues[5]);
    nn.feedforward();

    std::cout << nn.getAllOutput() << std::endl;

    return 0;
}