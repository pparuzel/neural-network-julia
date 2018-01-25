function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sigmoid_p(x)
    x = sigmoid(x)
    return x * (1 - x)
end

type Layer
    inputN::Int64
    outputN::Int64
    synapses::Array
    bias::Array
    inp::Float64
    net::Float64
    out::Float64
    act::Function
    act_p::Function
    feedforward::Function
    function Layer(inputN, outputN, isOutputLayer=False)
        this = new()

        this.inputN = inputN
        this.outputN = outputN
        # weights
        this.synapses = randn(inputN, outputN)
        # bias
        this.bias = isOutputLayer ? 0 : randn()
        # saved evaluation
        this.inp = nothing
        this.net = nothing
        this.out = nothing
        # activation function
        this.act = sigmoid
        this.act_p = sigmoid_p

        this.feedforward = function(data)
            this.inp = data
            data = data * this.synapses + this.bias
            this.net = data
            data = this.act(data)
            this.out = data
            return data
        end

        return this
    end
end

type NeuralNetwork
    η::Float64
    feedforward::Function
    function NeuralNetwork(
        input::Int64,
        hidden::Tuple,
        output::Int64,
        learning_rate::Float64)
        println("Starting Neural Network...")
        this = new()
        this.η = learning_rate
        println("Learning rate set to $(this.η)")

        this.feedforward = function(data)
            println("Passing $data forward...")
        end


        return this
    end
end

function main()
    nn = NeuralNetwork(2, (2,), 1, 0.01)
    nn.feedforward(10)
end


main()
