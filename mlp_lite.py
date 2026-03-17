import math

relu_fn = lambda x: max(0, x)
sigmoid_fn = lambda x: 1 / (1 + math.exp(x * -1))


class Perceptron:

    def __init__(self, id: str, weigths: list[float] | None = None, activation=relu_fn):
        self.id = id
        self.weigths = weigths
        self.activation = activation
        self.output: float | None = None
        self.inputs: list[float] = []
        if self.weigths is not None:
            self.inputs = [0 for _ in self.weigths]

    @property
    def signal(self) -> float:
        total = 0
        try:
            total = sum([input * weight for input, weight in zip(self.inputs, self.weigths)])
        except:
            raise ValueError(f"Bad number of inputs for signal {self.inputs} vs {self.weigths} on {self}")
        return total

    def propagation(self, inputs: list[float]) -> float:
        if len(inputs) == len(self.weigths) - 1:
            inputs.append(1)
        if len(inputs) != len(self.weigths):
            raise ValueError(f"Bad number of inputs {inputs} vs {self.weigths} on {self}")
        self.inputs = inputs
        self.output = self.activation(self.signal)
        return self.output

    def __repr__(self):
        s = f"P-{self.id} "
        for i, w in zip(self.inputs, self.weigths):
            s += f"+{i}*{w}"
        s += f"={self.signal}=>{self.output}"
        return s


class Layer:

    def __init__(self, id: str, nb_perceptron=0, activation=relu_fn):
        self.id = id
        self.perceptrons: list[Perceptron] = []
        for i in range(nb_perceptron):
            p = Perceptron(f"{id}-{i}", None, activation)
            self.perceptrons.append(p)

    def __repr__(self):
        return f"{self.id}:{len(self.perceptrons)}"


class MLP:

    def __init__(self, learning_rate=0.01):
        self.learningRate = learning_rate
        self.layers: list[Layer] = []

    def propagation(self, inputs: list[float]):
        for perceptron, input in zip(self.layers[0].perceptrons, inputs):
            perceptron.propagation([input])
        self._propagation()

    def _propagation(self):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i + 1]
            prev_layer = self.layers[i]
            for perceptron in layer.perceptrons:
                inputs = [p.output for p in prev_layer.perceptrons]
                perceptron.propagation(inputs)

    def outputs(self) -> list[float]:
        return [p.output for p in self.layers[-1].perceptrons]

    def predict(self, inputs: list[list[float]]) -> list[list[float]]:
        l = []
        for input in inputs:
            self.propagation(input)
            l.append(self.outputs())
        return l

    def __repr__(self):
        return f"MLP:{[l for l in self.layers]}"


if __name__ == '__main__':
    il = Layer("IL")
    for i in range(10):
        il.perceptrons.append(Perceptron(f"I{i}", [1]))

    hl = Layer("HL")
    hl.perceptrons.append(Perceptron("H0", [-0.18025137,-0.07142356, 0.07115833,0.25635878,0.59665951,0.56244433,-0.3020606, 0.05971639,-0.31818643,-0.16800152]))
    hl.perceptrons.append(Perceptron("H1", [-0.00212043,  0.27458788,0.32404,   0.22417334,0.19297939,0.14863863,-0.49707669,-0.43157449,0.19842946,0.84183263]))
    hl.perceptrons.append(Perceptron("H2", [-0.3898127,-0.32743939,-0.77163155,-0.14411409,-0.07917884,-0.50370773,-0.31211221,0.20759879,-0.13838611,-0.46663338]))
    hl.perceptrons.append(Perceptron("H3", [-0.3800719, 0.14319534,-0.17796858,-0.49656462,-0.06787931,0.16734378,-0.02849339,0.25038146,0.06631222,-0.0972953 ]))
    hl.perceptrons.append(Perceptron("H4", [-0.66572717,-0.03532317,0.46047437,-0.5097885, 0.54689601,-0.57808548,-0.66708663,-0.58417173,0.46607067,0.50158466]))
    hl.perceptrons.append(Perceptron("H5", [-0.58119109,0.54388772,0.81257123,0.03893687,0.41551034,-0.16613212,-0.73927856,-0.10773249,0.17993874,0.268476,]))
    hl.perceptrons.append(Perceptron("H6", [-0.22863389,-0.85476846,-0.34767259,-0.54431102,-0.9128886, 0.54616645, 0.2364781, 0.2954721, 0.30482541,-0.12674373]))
    hl.perceptrons.append(Perceptron("H7", [0.12176321,0.0484254,-0.08412969,  0.27351772,  0.09997187,-0.59858466,-0.06187847,-0.11426098,-0.80893849,-0.22543521]))

    # MLP
    ol = Layer("OL")
    ol.perceptrons.append(Perceptron("O0", [0.01280988, 0.48250077, -0.87647631, 0.63974548, 0.24925486, 0.35656178, -0.43275108, -0.77644924]))

    network = MLP()
    network.layers.append(il)
    network.layers.append(hl)
    network.layers.append(ol)
    print(network)

    data = [(28-4.775000000000000000e+01)/7.838207703295441142e+00,
            (1-7.115384615384615641e-01)/4.530468842072981062e-01,
            (2-2.951923076923077094e+00)/9.842979506334967876e-01,
            (130-1.320480769230769340e+02)/1.806365869730005258e+01,
            (132-2.496009615384615472e+02)/6.391181875051374561e+01,
            (0-6.730769230769230449e-02)/6.730769230769230449e-02,
            (2-2.548076923076922906e-01)/4.975671210407993650e-01,
            (185-1.403509615384615472e+02)/2.348561136915827063e+01,
            (0-2.884615384615384359e-01)/4.530468842072981062e-01,
            (0-5.735576923076922684e-01)/9.455476824298829630e-01]

    network.predict([data])

    print(network.outputs())
