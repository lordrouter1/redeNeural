#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "Neuron.cpp"

#ifndef __NEURALNETWORK__
#define __NEURALNETWORK__

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int nHidden, int hiddenSize, int outputSize, double learningRate)
        : learningRate(learningRate) {
        srand(static_cast<unsigned int>(time(nullptr)));

        // Inicialização dos neurônios e conexões ponderadas
        for (int i = 0; i < inputSize; ++i) {
            inputLayer.push_back(Neuron());
        }

        for(int i = 0; i < nHidden; ++i){
            std::vector<Neuron> hiddenLayerTemp;
            std::vector<std::vector<double>> hiddenWeightsTemp2;
            for (int j = 0; j < hiddenSize; ++j) {
                hiddenLayerTemp.push_back(Neuron());
                std::vector<double> hiddenWeightsTemp;
                if(j == 0){
                    for (int k = 0; k < inputSize; ++k) {
                        hiddenWeightsTemp.push_back((rand() % 2000 - 1000) / 1000.0);
                    }
                }
                else{
                    for (int k = 0; k < hiddenSize; ++k) {
                        hiddenWeightsTemp.push_back((rand() % 2000 - 1000) / 1000.0);
                    }
                }
                hiddenWeightsTemp2.push_back(hiddenWeightsTemp);
            }
            hiddenWeights.push_back(hiddenWeightsTemp2);
            hiddenLayer.push_back(hiddenLayerTemp);
        }

        for (int i = 0; i < outputSize; ++i) {
            outputLayer.push_back(Neuron());
            std::vector<double> outputWeightsTemp;
            for (int j = 0; j < hiddenSize; ++j) {
                outputWeightsTemp.push_back((rand() % 2000 - 1000) / 1000.0);
            }
            outputWeights.push_back(outputWeightsTemp);
        }
    }

    void setInput(const std::vector<double>& inputValues) {
        if (inputValues.size() == inputLayer.size()) {
            for (int i = 0; i < inputValues.size(); ++i) {
                inputLayer[i].setInput(inputValues[i]);
            }
        }
    }

    void feedforward() {
        int endLayer = 0;
        // Forward pass: calcular saída da rede usando conexões ponderadas
        for (int i = 0; i < hiddenLayer[0].size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputLayer.size(); ++j) {
                sum += inputLayer[j].getOutput() * hiddenWeights[0][i][j];
            }
            hiddenLayer[0][i].setInput(sum);
        }

        if(hiddenLayer.size() == 1){
            for (int i = 0; i < outputLayer.size(); ++i) {
                double sum = 0.0;
                for (int j = 0; j < hiddenLayer[0].size(); ++j) {
                    sum += hiddenLayer[0][j].getOutput() * outputWeights[i][j];
                }
                outputLayer[i].setInput(sum);
            }
        }
        else{
            for(int i = 0; i < hiddenLayer.size()-1; ++i){
                for(int j = 0; j < hiddenLayer[i+1].size(); ++j){
                    double sum = 0.0;
                    for(int k = 0; k < hiddenLayer[i+1].size(); ++k){
                        sum += hiddenLayer[i][k].getOutput() * hiddenWeights[i][j][k];
                    }
                    hiddenLayer[i+1][j].setInput(sum);
                }
            }
            
            endLayer = hiddenLayer.size()-1;
            for (int i = 0; i < outputLayer.size(); ++i) {
                double sum = 0.0;
                for (int j = 0; j < hiddenLayer[endLayer].size(); ++j) {
                    sum += hiddenLayer[endLayer][j].getOutput() * outputWeights[i][j];
                }
                outputLayer[i].setInput(sum);
            }
        }
    }

    void backpropagate(const std::vector<double>& targetValues) {
        const int lastHiddenLayer = hiddenLayer.size() - 1;
        
        // Calcular deltas para a camada de saída
        for (int i = 0; i < outputLayer.size(); ++i) {
            const double output = outputLayer[i].getOutput();
            const double error = targetValues[i] - output;
            const double delta = error * output * (1.0 - output);
            outputLayer[i].setDelta(delta);

            // Atualizar pesos entre camada oculta e camada de saída
            for (int j = 0; j < hiddenLayer[lastHiddenLayer].size(); ++j) {
                outputWeights[i][j] += learningRate * delta * hiddenLayer[lastHiddenLayer][j].getOutput();
            }
        }

        // Propagar deltas de volta através das camadas ocultas
        for (int layer = lastHiddenLayer; layer >= 0; --layer) {
            for (int i = 0; i < hiddenLayer[layer].size(); ++i) {
                const double output = hiddenLayer[layer][i].getOutput();
                double delta = 0.0;

                if (layer == lastHiddenLayer) {
                    // Para a última camada oculta, calcule o delta com base na camada de saída
                    for (int j = 0; j < outputLayer.size(); ++j) {
                        delta += outputLayer[j].getDelta() * outputWeights[j][i];
                    }
                } else {
                    // Para camadas ocultas anteriores, calcule o delta com base na próxima camada oculta
                    for (int j = 0; j < hiddenLayer[layer + 1].size(); ++j) {
                        delta += hiddenLayer[layer + 1][j].getDelta() * hiddenWeights[layer + 1][j][i];
                    }
                }

                delta *= output * (1.0 - output);
                hiddenLayer[layer][i].setDelta(delta);

                // Atualizar pesos entre camadas
                for (int j = 0; j < (layer == 0 ? inputLayer.size() : hiddenLayer[layer - 1].size()); ++j) {
                    double inputToUse = (layer == 0) ? inputLayer[j].getOutput() : hiddenLayer[layer-1][j].getOutput();

                    if (layer == 0) {
                        // Para a primeira camada oculta, atualize os pesos em relação à camada de entrada
                        hiddenWeights[layer][i][j] += learningRate * delta * inputToUse;
                    } else {
                        // Para camadas ocultas posteriores, atualize os pesos em relação à camada oculta anterior
                        hiddenWeights[layer][i][j] += learningRate * delta * inputToUse;
                    }
                }
            }
        }
    }

    void train(const std::vector<std::vector<double>>& inputSamples, const std::vector<std::vector<double>>& targetOutputs, int epochs, double target) {
        if (inputSamples.empty() || targetOutputs.empty()) {
            std::cerr << "Erro: Dados de entrada ou saída vazios." << std::endl;
            return;
        }

        if (inputSamples.size() != targetOutputs.size()) {
            std::cerr << "Erro: Tamanhos diferentes para dados de entrada e saída." << std::endl;
            return;
        }

        const int numSamples = inputSamples.size();
        const int numOutputs = outputLayer.size();
        double lastError = 2;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalError = 0.0;

            for (int sample = 0; sample < numSamples; ++sample) {
                // Define as entradas da rede neural
                setInput(inputSamples[sample]);

                // Executa o feedforward para calcular as saídas previstas
                feedforward();

                // Calcula o erro quadrático médio para este exemplo
                double exampleError = 0.0;
                for (int outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
                    const double targetValue = targetOutputs[sample][outputIdx];
                    const double predictedValue = outputLayer[outputIdx].getOutput();
                    exampleError += 0.5 * std::pow(targetValue - predictedValue, 2);
                }

                // Soma o erro deste exemplo ao erro total
                totalError += exampleError;

                // Executa o backpropagation para atualizar os pesos
                backpropagate(targetOutputs[sample]);
            }

            // Calcula a taxa média de erro para esta época
            const double meanError = totalError / numSamples;

            // Imprime o erro médio para esta época
            std::cout << "Epoca " << epoch + 1 << ", Erro Medio: " << meanError << std::endl;

            if(meanError > lastError) break;

            lastError = meanError;
        }
}

    double getOutput(int index) const {
        std::cout << "entrou\n";
        if (index >= 0 && index < outputLayer.size()) {
            return outputLayer[index].getOutput();
        }
        return 0.0;
    }

    std::string getAllOutput(){
        std::string resp = "";
        for(int i = 0; i < outputLayer.size();i++){
            resp += "["+std::to_string(outputLayer[i].getOutput())+"] ";
        }
        return resp;
    }

    void getWeights(){
        for(int i = 0; i < outputWeights.size(); i++){
            for(int j = 0; j < outputWeights[i].size();j++){
                std::cout << outputWeights[i][j] << " ";
            }
            std::cout  << std::endl;
        }
        std::cout  << std::endl;

        for(int i = 0; i < hiddenWeights.size(); i++){
            for(int j = 0; j < hiddenWeights[i].size(); j++){
                for(int k = 0; k < hiddenWeights[i][j].size();k++){
                    std::cout << hiddenWeights[i][j][k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }

private:
    std::vector<Neuron> inputLayer;
    std::vector<std::vector<Neuron>> hiddenLayer;
    std::vector<Neuron> outputLayer;

    std::vector<std::vector<std::vector<double>>> hiddenWeights;
    std::vector<std::vector<double>> outputWeights;
    
    double learningRate;
};

#endif