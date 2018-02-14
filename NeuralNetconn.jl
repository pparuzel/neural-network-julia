# Neural Network in Julia

module NeuralNetconn

export NeuralNetwork, predict, train, prepare, backup

#= Activation functions =#

sigmoid(x) = 1. / (1 + exp(-x))

function ∇sigmoid(x)
    x = sigmoid(x)
    return x * (1 - x)
end

LeakyReLU(x) = max(x, 0.01x)

∇LeakyReLU(x) = x > 0 ? 1 : 0.01

ELU(x) = max(x, 0.1 * (exp(x) - 1))

∇ELU(x) = max(1, 0.1 * (exp(x)))

function ∇tanh(x)
    x = tanh(x)
    return 1 - x * x
end

#= Neural Layer =#

mutable struct Layer
    # weights and biases
    synapses::Array
    bias::Array
    # activations
    out::Function
    ∇out::Function
    # saved values
    input::Array
    net::Array
    output::Array

    function Layer(ninput::Int, noutput::Int; isoutputlayer=false)
        synapses = randn(ninput, noutput)
        bias = isoutputlayer ? zeros(1, noutput) : randn(1, noutput)

        return new(synapses, bias, sigmoid, ∇sigmoid)
    end
end

#= Neural Network =#

mutable struct NeuralNetwork
    layers::Array{Layer}
    learning_rate::Float64

    function NeuralNetwork(ninput::Int64, nhidden::Tuple, noutput::Int64; η=0.01)
        l = [Layer(ninput, nhidden[1])]
        for i in 1:(length(nhidden) - 1)
            push!(l, Layer(nhidden[i], nhidden[i+1]))
        end
        push!(l, Layer(nhidden[end], noutput, isoutputlayer=true))
        return new(l, η)
    end

    function NeuralNetwork(dumpdata)
        return new(dumpdata["layers"], dumpdata["lr"])
    end
end

#= Feedforward algorithm =#
# assigns 3 values to every layer
# input, linear regression output
# and post-activation output

function feedforward(data::Array, nn::NeuralNetwork)
    for layer in nn.layers
        layer.input = data
        data = data * layer.synapses .+ layer.bias
        layer.net = data
        data = layer.out.(data)
        layer.output = data
    end
    return data
end

err(prediction, target) = 0.5 * (target - prediction) ^ 2

∇err(prediction, target) = prediction - target

function backpropagate(output::Array, nn::NeuralNetwork)
    η = nn.learning_rate
    # output layer
    layer = nn.layers[end]
    # calculate partial derivatives
    ∂err_∂out = ∇err.(layer.output, output)
    ∂out_∂net = layer.∇out.(layer.net)
    # save δᵢⱼ
    δ = ∂err_∂out .* ∂out_∂net
    # update weights
    layer.synapses += -η * layer.input' * δ
    # hidden layers
    for i in length(nn.layers)-1:-1:1
        layer = nn.layers[i]
        ∂err_∂out = δ * nn.layers[i + 1].synapses.'
        ∂out_∂net = layer.∇out.(layer.net)
        # save δᵢⱼ
        δ = ∂err_∂out .* ∂out_∂net
        # println("δ: $(δ)")
        # update weights
        layer.synapses += -η * layer.input.' * δ
        # update bias
        layer.bias = layer.bias .+ (-η * δ)
    end
end

# prepare data for feed
function prepare(data)::Array{Float64, 2}
    data = cat(2, data)
    return data
end

function predict(data::Array, output::Array, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn)
    # calculate cost vector
    cost = err.(result, output)
    # get the sum of costs
    cost = sum(cost)
    return Dict("result" => result, "cost" => cost)
end

function train(data::Array, output::Array, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn)
    # calculate cost vector
    cost = err.(result, output)
    # get the sum of costs
    cost = sum(cost)
    # learn
    backpropagate(output, nn)
    return Dict("result" => result, "cost" => cost)
end

function predict(data::Array, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn)
    return Dict("result" => result)
end

function backup(nn)
    return Dict("layers" => nn.layers, "lr" => nn.learning_rate)
end

end

nothing
