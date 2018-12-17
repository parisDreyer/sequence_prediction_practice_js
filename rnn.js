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
        this.weights = Layer.init_weights(input_size);
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