using NeuralNetconn
include("mnist.jl")

data = mnist()

nn = NeuralNetwork(784, (28,), 10, η=1)
costs = zeros(60000)

# prepare input data
flatdata = reshape(data[1], (784, 1, 60000))
# prepare output data
wideans = zeros(UInt8, (60000, 10))
for i in 1:60000
    # put 1 on position from 1 to 10
    # 10 equals 0 in this MNIST case
    wideans[i, data[2][i]] = 0x01
end

# training loop
function training(iters=3)
    for it in 1:iters
        for i in 1:60000
            if i % 500 == 0
                print(STDERR, "\r Training: $(div(i, 600))%")
            end
            result = train(prepare(flatdata[:, 1, i]'), prepare(wideans[i, :]'), nn)
            costs[i] = result["cost"]
        end
        println("\nOverall error: ", sum(costs))
    end
end

# check how many predictions is valid
function check()
    checks = Dict("HIT" => 0, "MISS" => 0)
    for i in 1:60000
        if i % 500 == 0
            print(STDERR, "\rChecking: $(div(i, 600))%")
        end
        if UInt8(indmax(predict(prepare(flatdata[:, 1, i]'), prepare(wideans[i, :]'), nn)["result"])) == data[2][i]
            checks["HIT"] += 1
        else
            checks["MISS"] += 1
        end
    end
    println("\nAccuracy: $(checks["HIT"] / 600)%")
    return checks
end

# training()
# check()
