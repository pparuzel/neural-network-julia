using NeuralNetconn

# create data
dataset = [0 0; 0 1; 1 0; 1 1]
answers = [0; 1; 1; 0]
# prepare data
dataset = prepare(dataset)
answers = prepare(answers)
# initialize net
nn = NeuralNetwork(2, (4, 4), 1, η=0.01)

function training()
    for i in 1:10000
        ri = rand(1:4)
        train(prepare(dataset[ri, :].'), prepare(answers[ri].'), nn)
    end
end

function predictions()
    for i in 1:4
        println( "$(dataset[i, :])\t$(predict(prepare(dataset[i, :].'), nn))" )
    end
end

foo(x) = 0.000000005x^2 + 0.01

function progressive(iters=10000)
    for i in 1:iters
        ri = rand(1:4)
        train(prepare(dataset[ri, :].'), prepare(answers[ri].'), nn)
        nn.learning_rate = foo(iters)
        if i % 100 == 0
            @printf(STDERR, "\r [%.6f] [%.6f] [%.6f] [%.6f]",
                predict([0. 0.], nn)[1],
                predict([0. 1.], nn)[1],
                predict([1. 0.], nn)[1],
                predict([1. 1.], nn)[1])
        end
    end
    @printf("\n %.2f %.2f %.2f %.2f",
            round(predict([0. 0.], nn)[1], 1),
            round(predict([0. 1.], nn)[1], 1),
            round(predict([1. 0.], nn)[1], 1),
            round(predict([1. 1.], nn)[1], 1))
end
