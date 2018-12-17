const { NNMath } = require('./utils/neuralnet_math.js');
class Model{

    constructor({input_size, layers, temperature }){

        this.input_size = input_size;
        this.layers = layers.map((size, i) =>{
            if(i === 0){
                return new Layer({ input_size, size, temperature });
            } else {
                return new Layer({ input_size: layers[i - 1], size, temperature})
            }
        });
        this.past_ten_deltas = [];
        this.past_ten_target_outputs = [];
        this.prev_output = undefined;
    }

    accuracy(){
        if(this.past_ten_deltas.length === 0) return 0;

        let errs = this.past_ten_deltas[0];
        let avg_targets = this.past_ten_target_outputs[0];

        for(let i = 1; i < this.past_ten_deltas.length; ++i){
            for(let j = 0; j < this.past_ten_deltas[i].length; ++j){
                errs[j] += this.past_ten_deltas[i][j];
                avg_targets[j] += this.past_ten_target_outputs[i][j]
            }
        }
        errs = errs.map((es, i) => // get the ratios
            (es / this.past_ten_deltas.length) / avg_targets[i]);

        return NNMath.sum(errs) / errs.length; // return the accuracy
    }

    predict(input){
        let output = this.layers[0].predict(input);
        for(let i = 1; i < this.layers.length; ++i){
            output = this.layers[i].predict(output);
        }

        this.prev_output = output;
        return output;
    }


    adjust(target_output){
        let i = this.layers.length - 1;
        
        let nxt_err = this.layers[i].adjust(target_output);
        this.update_past_ten(target_output, this.layers[i].prev_output_error)

        i -= 1;
        while(i >= 0){
            nxt_err = this.layers[i].adjust(nxt_err);
            i -= 1;
        }
    }

    update_past_ten(target, error){
        if(this.past_ten_deltas.length > 15){
            this.past_ten_deltas = this.past_ten_deltas.slice(5);
            this.past_ten_target_outputs = this.past_ten_target_outputs.slice(5);
        } 
        this.past_ten_deltas.push(error);
        this.past_ten_target_outputs.push(target);
    }

}


class Layer{

    constructor({ input_size, size, temperature }) {
        this.input_size = input_size;
        this.size = size;


        this.prev_input = undefined;
        this.input_error = undefined;
        this.prev_output = undefined;
        this.prev_output_error = undefined; // used for calculating accuracy

        this.nodes = [];
        for(let i = 0; i < size; ++i){
            this.nodes.push(new Perceptron({ input_size, size, temperature }));
        }

    }

    predict(input){
        this.prev_input = input;
        if(this.size === 0) return input;

        let output = this.nodes.map(n => n.predict(input));

        this.prev_output = output;
        return output;
    }

    adjust(output_error){
        let deltas = output_error.map((err, i) => this.prev_output[i] - err);
        this.prev_output_error = deltas;
        if(this.size === 0) return deltas;


        let inpt_errs = this.nodes[0].adjust(deltas[0]);
        for(let i = 1; i < this.nodes.length; ++i){
            inpt_errs = NNMath.vals_plus_vals(
                inpt_errs, 
                this.nodes[i].adjust(deltas[i]));
        }
        this.input_error = inpt_errs;

        return inpt_errs;
    }

}



class Perceptron {
    constructor({input_size, size, temperature }){
        this.input_size = input_size;
        this.size = size;

        this.prev_input = undefined;
        this.prev_output = undefined;
        this.temperature = temperature || 0.01;

        this.weights = Perceptron.init_weights(input_size);
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

        let err_signal = NNMath.tanh_deriv(this.prev_output - output_error);
        this.input_errors = this.weights.map(w => w * err_signal);
        // adjust the weights
        for(let i = 0; i < this.weights.length; ++i){
            this.weights[i] += this.temperature * this.input_errors[i];
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