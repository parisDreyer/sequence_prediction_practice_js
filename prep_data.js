/**
 * https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array/6274398
 * Shuffles array in place.
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
    let j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}


// groups array els into array of arrays of six els
function groups_of_six(a) {
    let groups = [];
    let sub = a.slice();
    while(sub.length > 0){
        let nxt_arr = []
        for(let i = 0; i < 6; ++i){
            nxt_arr.push(sub.pop());
        }
        // reverse
        groups.push(nxt_arr.reverse());

    }

    return groups;
}


// min and max included
function rand(min, max) {
    if(min === max) return min || 0; 
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// no error checking lazy coding because this is for one project
// flattens an array once
function flatten_once(a){
    let flattened = [];
    for(let i = 0; i < a.length; ++i){
        let nxts = [];
        while(a[i].length > 0){
            nxts.push(a[i].pop());
        }
        // reverse nxts
        while(nxts.length > 0){
            flattened.push(nxts.pop());
        }
    }
    return flattened;
}

// return arrays shuffled by every six
// assume a.length is evenly divisible by 6
function prep_rnn_input(a){
    let arrs = shuffle(groups_of_six(a))
    return flatten_once(arrs);
}

module.exports = { prep_rnn_input };