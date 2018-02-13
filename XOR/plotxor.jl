include("xor.jl")
using Plots; gr()

# Create 100 images of the plot
# using progressive learning rate
function gifxor(iters=10000)
    foo(x) = min(0.000000005x^2 + 0.01, 0.51)
    # setup x-axis and y-axis
    x = y = linspace(0., 1., 100)
    # main loop
    for i in 1:iters
        if i % 100 == 0
            contour(x, y, (x, y) -> predict([x y], nn)[1], fill=false, c=:viridis)
            png("gif/nn$(Int64(round(i/100)))")
        end
        ri = rand(1:4)
        train(prepare(dataset[ri, :].'), prepare(answers[ri].'), nn)
        nn.learning_rate = foo(iters)
    end
end

# Plot XOR problem
function plotxor()
    global nn
    x = y = linspace(0., 1., 100)
    contour(x, y, (x, y) -> predict([x y], nn)[1], fill=true, c=:viridis)
end
