const { NNMath } = require('./utils/neuralnet_math.js');
class Model{

    constructor({input_size, layers, temperature, last_output }){
        last_output = last_output || 'tanh';

        this.input_size = input_size;
        this.layers = layers.map((size, i) =>{
            if(i === 0){
                return new Layer({ input_size, size, temperature });
            } else if (i < layers.length - 1){
                return new Layer({ input_size: layers[i - 1], size, temperature})
            } else {
                return new Layer({ input_size: layers[i - 1], size, temperature, last_output })
            }
        });
        this.past_ten_deltas = [];
        this.past_ten_target_outputs = [];
        this.past_ten_outputs = [];
        this.prev_output = undefined;
    }


    truncate(){
        // truncates weights to needed decimal precision
        for(let i = 0; i < this.layers.length; ++i){
            this.layers[i].truncate();
        }
    }

    accuracy(){
        let sample_size  = this.past_ten_deltas.length; // num samples taken

        if(sample_size === 0) return 0;

        // sumed standard deviation
        let sum_err_std = this.past_ten_deltas[0].map(e => e*e);
        let sum_targets_std = this.past_ten_target_outputs[0];
        for(let i = 1; i < sample_size; ++i){

            for(let j = 0; j < this.past_ten_deltas[i].length; ++j){

                let e = this.past_ten_deltas[i][j]; // target
                let t = this.past_ten_target_outputs[i][j]; // target
                sum_err_std[j] += Math.sqrt(e * e); // square root of the variance
                sum_targets_std[j] += Math.sqrt(t * t);
            }
        }

        // average standard deviation
        let std = sum_err_std.map(e => e / sample_size); // average standard deviation of error
        let mean = sum_targets_std.map(total => total / sample_size); // average std of target
        

        // the summed difference between the error and the target relative to the target
        let sum_acc = 0;
        for(let i = 0; i < std.length; ++i){
            // the lower the standard deviation the more accurate the model!
            // because the closer the below operation gets to 1/1
            sum_acc += 1 - Math.abs(1 - Math.abs(mean[i] - std[i]) / mean[i]); 
        
        }
        return sum_acc / std.length;
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

        this.update_past_ten(
            this.prev_output, 
            target_output, 
            this.layers[i].prev_output_error);

        i -= 1;
        while(i >= 0){
            nxt_err = this.layers[i].adjust(nxt_err);
            i -= 1;
        }
    }

    update_past_ten(output, target, error){
        if(this.past_ten_deltas.length > 25){
            this.past_ten_deltas = this.past_ten_deltas.slice(10);
            this.past_ten_target_outputs = this.past_ten_target_outputs.slice(10);
            this.past_ten_outputs = this.past_ten_target_outputs.slice(10);
        } 
        this.past_ten_deltas.push(error);
        this.past_ten_target_outputs.push(target);
        this.past_ten_outputs.push(output);
    }

}


class Layer {
  constructor({ input_size, size, temperature, last_output }) {
    this.input_size = input_size;
    this.size = size;

    this.prev_input = undefined;
    this.input_error = undefined;
    this.prev_output = undefined;
    this.prev_output_error = undefined; // used for calculating accuracy

    this.nodes = [];
    for (let i = 0; i < size; ++i) {
      this.nodes.push(new Perceptron({ input_size, size, temperature, last_output }));
    }
  }

  truncate(){ // truncates weights to reasonable decimal precision
    for(let i = 0; i < this.size; ++i){
        this.nodes[i].truncate();
    }
  }
  predict(input) {
    this.prev_input = input;
    if (this.size === 0) return input;

    let output = this.nodes.map(n => n.predict(input));

    this.prev_output = output;
    return output;
  }

  adjust(output_error) {
    let deltas = output_error.map((err, i) => this.prev_output[i] - err);
    this.prev_output_error = deltas;
    if (this.size === 0) return deltas;

    let inpt_errs = this.nodes[0].adjust(deltas[0]);
    for (let i = 1; i < this.nodes.length; ++i) {
      inpt_errs = NNMath.vals_plus_vals(
        inpt_errs,
        this.nodes[i].adjust(deltas[i])
      );
    }
    this.input_error = inpt_errs;

    return inpt_errs;
  }
}



class Perceptron {
    constructor({input_size, size, temperature, last_output }){
        this.input_size = input_size;
        this.size = size;

        this.prev_input = undefined;
        this.pre_activation_output = undefined;
        this.prev_output = undefined;
        this.temperature = temperature || 0.01;

        this.weights = Perceptron.init_weights(input_size);
        this.input_errors = undefined; // errors with respect to weights


        this.last_output = last_output; // tanh or linear or softmax ....
    }

    // http://www.jacklmoore.com/notes/rounding-in-javascript/
    truncate(){ 
        // trucates weights to shorten calculation time
        // precise to 8 decimal places (can't count on the ninth digit for precision)
        this.weights = this.weights.map(w =>
            Math.round(w * 1000000000) / 1000000000.0
          ) ;
    }


    predict(input) {
        this.prev_input = input;
        let output;
        if(this.last_output === 'linear'){ // linear
            this.pre_activation_output = NNMath.sum(
                        NNMath.vals_times_vals(this.weights, input));
            output = this.pre_activation_output;
        }else{ // tanh
            this.pre_activation_output = 
                    NNMath.sum(
                        NNMath.vals_times_vals(this.weights, input));
            output = NNMath.fast_tanh(this.pre_activation_output);
        }
        this.prev_output = output;

        return output;
    }

    adjust(output_error){

        let err_signal;
        if (this.last_output === 'linear') { // linear
            err_signal = output_error;
        }else { // tanh derivative
            err_signal = NNMath.tanh_deriv(this.pre_activation_output) * output_error;
        }
        this.input_errors = this.weights.map(w => w * err_signal);
        // adjust the weights
        for(let i = 0; i < this.weights.length; ++i){
            this.weights[i] += this.temperature * this.input_errors[i] * this.prev_input[i];
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