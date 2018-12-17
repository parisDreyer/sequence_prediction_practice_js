import { Numbers } from './res/numslist.js'; // the data
import { Model } from './rnn.js';
import { prep_rnn_input } from './prep_data.js'; // does some specific shuffling for an array of arrays


// desired accuracy
const target_accuracy = 0.99; // 99%


// io
const input_size = 5; // array of  5 values
const output_size = 5;
const max_epochs = 300;

// layers and nodes per layer
const nodes1 = 20;
const nodes2 = 20;
const nodes3 = 20;
const layers = [
    nodes1,
    nodes2,
    nodes3
];

let tha_model = new Model({
    input_size,
    layers,
    output_size
});


function train_model(model, target_accuracy, max_epochs){

    let current_epoch = 0;

    while(model.accuracy < target_accuracy || current_epoch > max_epochs){

        let new_samples = prep_rnn_input(Numbers);

        for(let i = 0; i < new_samples.length; i += 6){


            for(let s = i; s < i + 5; s++){

                model.predict(new_samples[s]);
                model.adjust(new_samples[s + 1]); // adjust to next in sequence
            }
        }

        current_epoch++;
    }

    return model;
}





// train the model
let trained_model = train_model(tha_model, target_accuracy, max_epochs);

// predict next number sequence
trained_model.predict(Numbers[Numbers.length - 1]); 

