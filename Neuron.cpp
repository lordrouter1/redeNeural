#include <cmath>

#ifndef __NEURON__
#define __NEURON__ 1

class Neuron
{
    public:
        Neuron() : input(0), output(0), delta(0) {}

        void setInput(double input) {
            this->input = input;
        }

        double getOutput() const {
            return sigmoid(input);
        }

        double getDelta() const {
            return delta;
        }

        void setDelta(double delta) {
            this->delta = delta;
        }

    private:
        double input;
        double output;
        double delta;
        
        double sigmoid(double x) const {
            return 1.0 / (1.0 + exp(-x));
        }

        double tanh(double x) const {
            return std::tanh(x);
        }

        double relu(double x) const{
            return std::max(0.0, x);
        }

        double hardTanh(double x) const{
            return std::max(-1.0, std::min(1.0, x));
        }
};

#endif