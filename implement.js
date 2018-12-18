const { Numbers } = require('./res/numslist.js'); // the data
const { Model } = require('./nn.js');
const { prep_rnn_input, print_to_file } = require('./prep_data.js'); // does some specific shuffling for an array of arrays


// desired accuracy
const target_accuracy = 0.99; // 99%


// io
const input_size = 5; // array of  5 values
const output_size = 5;
const max_epochs = 20000;

// layers and nodes per layer

const layers = [
    256, // first layer
    42,
    256,
    output_size // output size
];

let tha_model = new Model({ input_size, layers, temperature: 0.0001, last_output: 'linear' });


async function train_model(model, target_accuracy, max_epochs){

    let current_epoch = 0;

    let accuracy = 0;
    while(accuracy < target_accuracy && current_epoch < max_epochs){

        let new_samples = prep_rnn_input(Numbers);
        for(let i = 0; i < new_samples.length; i += 6){


            for(let s = i; s < i + 5; s++){

                model.predict(new_samples[s]);
                model.adjust(new_samples[s + 1]); // adjust to next in sequence
            }
        }

        current_epoch++;

        model.truncate(); // truncate weights to needed decimal precision
        accuracy = model.accuracy();
        console.log("sample output: ", model.predict(new_samples[new_samples.length - 1]))
        console.log("accuracy: ", accuracy);
    }

    return model;
}







train_model(tha_model, target_accuracy, max_epochs).then(m => {
    // train the model
    let trained_model = m;
    // predict next number sequence
    let prediction = trained_model.predict(Numbers[Numbers.length - 1]);
    console.log("THE PREDICTION: ",
        JSON.stringify(prediction)); 
    if(!!prediction[0]) print_to_file(trained_model);
});




