const { Numbers } = require('./res/numslist.js'); // the data
const { Model } = require('./nn.js');
const { prep_rnn_input } = require('./prep_data.js'); // does some specific shuffling for an array of arrays


// desired accuracy
const target_accuracy = 0.99; // 99%


// io
const input_size = 5; // array of  5 values
const output_size = 5;
const max_epochs = 60000;

// layers and nodes per layer
const nodes1 = 20;
const nodes2 = 20;
const nodes3 = output_size;
const layers = [
    nodes1,
    nodes2,
    nodes3 // output size
];

let tha_model = new Model({ input_size, layers });


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


        accuracy = model.accuracy();
        console.log("accuracy: ", accuracy);
    }

    return model;
}







train_model(tha_model, target_accuracy, max_epochs).then(m => {
    // train the model
    let trained_model = m;
    // predict next number sequence
    console.log("THE PREDICTION: ",
        JSON.stringify(trained_model.predict(Numbers[Numbers.length - 1]))); 

});


