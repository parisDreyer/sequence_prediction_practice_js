

class NNMath{

    // between 0 and 0.05
    static rand_weights(max = 0.0500000001, min = 0.0) {
        return (Math.random() * (max - min) + min);
    }


    static fast_tanh(x) {

        // return Math.tanh(x);
        if (x < -3)
            return -1;
        else if (x > 3)
            return 1;
        else
            return x * (27 + x * x) / (27 + 9 * x * x);
    }

    static tanh_deriv(x){
        let ft = NNMath.fast_tanh(x);
        return 1 - (ft * ft);
    }


    static vals_times_vals(vals1, vals2){
        
        if(vals1.length != vals2.length)
            throw "vals must be same length";
        let res = [];
        for(let i = 0; i < vals1.length; ++i){
            res.push(vals1[i] * vals2[i]);
        }

        return res;
    }


    static vals_plus_vals(vals1, vals2) {

        if (vals1.length != vals2.length)
            throw "vals must be same length";
        let res = [];
        for (let i = 0; i < vals1.length; ++i) {
            res.push(vals1[i] + vals2[i]);
        }

        return res;
    }

    // min and max included
    static rand(min, max) {
        if (min === max) return min || 0;
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    static sum(a){
        let total = 0;
        for(let i = 0; i < a.length; ++i){
            total += a[i];
        }
        return total;
    }
}

module.exports = { NNMath };