import { NNMath } from 'utils/neuralnet_math.js';
class Model{

    constructor({input_size, layers, output_size }){

        this.input_size = input_size;
        this.layers = layers.map((size, i) =>{
            if(i === 0){
                return new Layer({ input_size, size });
            } else {
                return new Layer({ input_size: layers[i - 1], size})
            }
        })
        this.accuracy = 0;
    }


}



class Layer {
    constructor({input_size, size}){
        this.input_size = input_size;
        this.size = size;

        this.prev_input = undefined;
        this.prev_output = undefined;

        this.weights = Layer.init_weights(input_size);
        this.input_errors = undefined; // errors with respect to weights
    }


    predict(input) {
        this.prev_input = input;
        
        let output = 
            NNMath.fast_tanh(
                NNMath.sum(
                    NNMath.vals_times_vals(this.weights, input)));
        this.prev_output = output;

        return output;
    }

    adjust(output_error){
        let err_signal = this.prev_output - NNMath.tanh_deriv(output_error);
        this.input_errors = this.weights.map(w => w * err_signal);
        // adjust the weights
        for(let i = 0; i < this.weights.length; ++i){
            this.weights[i] -= 0.0002 * this.input_errors[i];
        }
        
        return this.input_errors;
    }



    static init_weights(num){
        let weights = [];
        for(let i = 0; i < num; ++i){
            weights.push(NNMath.rand_weights());
        }
        return weights;
    }
}





module.exports = { Model }