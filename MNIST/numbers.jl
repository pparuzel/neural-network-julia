using SFML

include("mnist_loader.jl")
include("nn.jl")

training()

window = RenderWindow(VideoMode(800, 600), "Numbers")
set_framerate_limit(window, 60)

data = mnist()

function get_sprite(n::Int64)
    img = Image(28, 28)
    colors = data[1][:, :, 1, n] * 255
    for i in 1:28
        for j in 1:28
            clr = convert(UInt8, colors[i, j])
            set_pixel(img, i, j, Color(clr, clr, clr))
        end
    end
    spr = Sprite(Texture(img))
    set_scale(spr, Vector2f(10.0, 10.0))
    set_position(spr, Vector2f(260, 130))
    return spr
end

text = RenderText()
set_position(text, Vector2f(10.0, 10.0))
set_string(text, "Prediction: ")
set_style(text, TextBold)
set_color(text, SFML.white)
set_charactersize(text, 50)


function main()
    counter = 0
    index = 1
    currentsprite = get_sprite(1)
    prediction = UInt8(indmax(predict(prepare(flatdata[:, 1, 1]'), nn)["result"]))
    event = Event()
    while isopen(window)
        while pollevent(window, event)
            if get_type(event) == EventType.CLOSED
                close(window)
            end
        end
        clear(window, SFML.black)
        draw(window, currentsprite)
        counter += 1
        if counter % 60 == 0
            counter = 0
            index += 1
            if index > 60000
                close(window)
            end
            currentsprite = get_sprite(index)
            prediction = UInt8(indmax(predict(prepare(flatdata[:, 1, index]'), nn)["result"]))
            prediction = prediction == 10 ? 0 : prediction
            set_string(text, "Prediction: $(prediction)")
        end
        draw(window, text)

        display(window)
    end

end
