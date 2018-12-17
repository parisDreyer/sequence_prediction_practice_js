

class NNMath{

    // between -0.02 and 0.02
    static rand_weights() {
        return (Math.random() * (-0.0200 - 0.0200) + 0.0200);
    }


    static fast_tanh(x) {
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


    static sum(a){
        let total = 0;
        for(let i = 0; i < a.length; ++i){
            total += a[i];
        }
        return total;
    }
}

module.exports = { NNMath };