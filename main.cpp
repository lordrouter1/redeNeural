#include "NeuralNetwork.cpp"

int main()
{
    std::ifstream inp("in");
    std::vector<std::vector<double>> inputValues;
    std::vector<std::vector<double>> targetValues;
    std::string linha;

    int numEntrada;
    int numCamada;
    int qtdCamada;
    int numSaida;
    double txAprend;

    inp >> numEntrada >> numCamada >> qtdCamada >> numSaida >> txAprend;

    NeuralNetwork nn(numEntrada, numCamada, qtdCamada, numSaida, txAprend);

    int voltas = 0;

    std::getline(inp,linha);
    while(std::getline(inp,linha)){
        int cont = 0;
        std::vector<double> temp[2];
        for(int i = 0; i < linha.size(); i ++){
            if(linha[i] == '|'){cont++;}
            else if(linha[i] != ' '){
                temp[cont].push_back(linha[i]);
            }
        }
        inputValues.push_back(temp[0]);
        targetValues.push_back(temp[1]);
    }
    std::cout << inputValues.size() << "\t" << inputValues[0].size() << std::endl;
    std::cout << targetValues.size() << "\t" << targetValues[0].size() << std::endl;
    
    /*
    if(true){
        nn.train(inputValues,targetValues,20000000,0.00000001);
        nn.getWeights();
    }
    else{
        nn.importDeltas("deltas.out");
        nn.importWeights("weights.out");
    }

    nn.setInput(inputValues[5]);
    nn.feedforward();

    std::cout << nn.getAllOutput() << std::endl;
    */

    return 0;
}