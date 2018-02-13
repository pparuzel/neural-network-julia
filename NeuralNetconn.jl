# Neural Network in Julia

module NeuralNetconn

export NeuralNetwork, predict, train, prepare

#= Activation functions =#

sigmoid(x::Float64) = 1. / (1 + exp(-x))

function ∇sigmoid(x::Float64)
    x = sigmoid(x)
    return x * (1 - x)
end

LeakyReLU(x::Float64) = max(x, 0.01x)

∇LeakyReLU(x::Float64) = x > 0 ? 1 : 0.01

ELU(x::Float64) = max(x, 0.1 * (exp(x) - 1))

∇ELU(x::Float64) = max(1, 0.1 * (exp(x)))

function ∇tanh(x::Float64)
    x = tanh(x)
    return 1 - x * x
end

#= Neural Layer =#

type Layer
    # weights and biases
    synapses::Array{Float64}
    bias::Array{Float64}
    # activations
    out::Function
    ∇out::Function
    # saved values
    input::Array{Float64}
    net::Array{Float64}
    output::Array{Float64}

    function Layer(ninput::Int, noutput::Int; isoutputlayer=false)
        synapses = randn(ninput, noutput)
        bias = isoutputlayer ? zeros(1, noutput) : randn(1, noutput)

        return new(synapses, bias, sigmoid, ∇sigmoid)
    end
end

#= Neural Network =#

type NeuralNetwork
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
end

#= Feedforward algorithm =#
# assigns 3 values to every layer
# input, linear regression output
# and post-activation output

function feedforward(data::Array{Float64}, nn::NeuralNetwork)
    for layer in nn.layers
        layer.input = data
        data = data * layer.synapses .+ layer.bias
        layer.net = data
        data = layer.out.(data)
        layer.output = data
    end
    return data
end

err(prediction::Float64, target::Float64) = 0.5 * (target - prediction) ^ 2

∇err(prediction::Float64, target::Float64) = prediction - target

function backpropagate(output::Array{Float64}, nn::NeuralNetwork)
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

function predict(data::Array{Float64}, output::Array{Float64}, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn)
    # calculate cost vector
    cost = err.(result, output)
    # get the sum of costs
    cost = sum(cost)
    return Dict("result" => result, "cost" => cost)
end

function train(data::Array{Float64}, output::Array{Float64}, nn::NeuralNetwork)
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

function predict(data::Array{Float64}, nn::NeuralNetwork)
    # propagate forward
    result = feedforward(data, nn)
    return result
end

end
